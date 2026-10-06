"""Standalone state-adaptive Kaggriculture agent."""

from __future__ import annotations

import base64
import copy
import io
import json
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
ITEMS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER", *ANIMALS)

@dataclass(frozen=True, slots=True)
class TapeSelectorConfig:
    branch_step: int = 145
    neighbors: int = 3
    distance_epsilon: float = 1e-6
    ood_threshold: float = float("inf")
    risk_coefficient: float = 0.0

@dataclass(frozen=True, slots=True)
class TapeSelectorCheckpoint:
    config: TapeSelectorConfig
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    train_features: np.ndarray
    q_values: np.ndarray
    global_action: int
    action_ids: tuple[int, ...]
    metadata: dict[str, Any]

def _tile_features(farm: dict[str, Any]) -> list[float]:
    kinds: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    ready = 0
    needs_water = 0
    fertilizer = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "EMPTY"))
            kinds[kind] += 1
            crop = tile.get("crop")
            if crop is not None:
                crops[str(crop)] += 1
            animal = tile.get("animal")
            if animal is not None:
                animals[str(animal)] += 1
            ready += int(float(tile.get("yield_units", 0) or 0) > 0)
            needs_water += int(kind == "PLANT" and not tile.get("watered_today", False))
            fertilizer += int(float(tile.get("fertilizer_available", 0) or 0) > 0)

    return [
        len(farm.get("hands", [])) / 16.0,
        len(farm.get("unlocked_quadrants", [])) / 4.0,
        float(farm.get("hires_today", 0) or 0) / 16.0,
        kinds["WEED"] / 100.0,
        kinds["PASTURE"] / 30.0,
        kinds["COOP"] / 30.0,
        ready / 30.0,
        needs_water / 30.0,
        fertilizer / 30.0,
        *(crops[crop] / 30.0 for crop in CROPS),
        *(animals[animal] / 30.0 for animal in ANIMALS),
    ]

def encode_selector_features(observation: dict[str, Any]) -> np.ndarray:
    """Encode only information available to the candidate at decision time."""
    player = int(observation.get("player", 0))
    opponent = 1 - player
    farms = observation["farms"]
    candidate_farm = farms[player]
    opponent_farm = farms[opponent]
    candidate_money = float(candidate_farm.get("money", 0) or 0)
    opponent_money = float(opponent_farm.get("money", 0) or 0)
    prices = observation.get("market", {}).get("prices", {})
    private = observation.get("private", {})
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    inventory: Counter[str] = Counter()
    for unit_inventory in private.get("inventories", []):
        inventory.update(
            {
                str(item): int(quantity)
                for item, quantity in unit_inventory.items()
            }
        )

    features = [
        float(observation.get("step", 0) or 0) / 719.0,
        float(observation.get("day", 0) or 0) / 30.0,
        float(observation.get("hour", 0) or 0) / 24.0,
        candidate_money / 200_000.0,
        opponent_money / 200_000.0,
        (candidate_money - opponent_money) / 200_000.0,
        *(float(prices.get(item, 0) or 0) / 2_000.0 for item in ITEMS),
        *_tile_features(candidate_farm),
        *_tile_features(opponent_farm),
        *(float(shed.get(item, 0) or 0) / 200.0 for item in ITEMS),
        *(float(seeds.get(crop, 0) or 0) / 50.0 for crop in CROPS),
        *(float(inventory[item]) / 100.0 for item in ITEMS),
    ]
    return np.asarray(features, dtype=np.float64)

def predict_q(
    checkpoint: TapeSelectorCheckpoint, features: np.ndarray
) -> tuple[np.ndarray, float]:
    normalized = (features - checkpoint.feature_mean) / checkpoint.feature_scale
    distances = np.linalg.norm(checkpoint.train_features - normalized, axis=1)
    count = min(checkpoint.config.neighbors, len(distances))
    nearest = np.argpartition(distances, count - 1)[:count]
    minimum_distance = float(distances[nearest].min())
    if minimum_distance > checkpoint.config.ood_threshold:
        values = np.full(checkpoint.q_values.shape[1], -np.inf)
        values[checkpoint.global_action] = 0.0
        return values, minimum_distance
    weights = 1.0 / (distances[nearest] + checkpoint.config.distance_epsilon)
    neighbor_values = checkpoint.q_values[nearest]
    mean_values = np.average(neighbor_values, axis=0, weights=weights)
    centered = neighbor_values - mean_values
    variance = np.average(np.square(centered), axis=0, weights=weights)
    values = mean_values - checkpoint.config.risk_coefficient * np.sqrt(variance)
    return values, minimum_distance

def select_action(
    checkpoint: TapeSelectorCheckpoint, features: np.ndarray
) -> tuple[int, np.ndarray, float]:
    values, distance = predict_q(checkpoint, features)
    column = int(np.argmax(values))
    return checkpoint.action_ids[column], values, distance

_CHECKPOINT_BLOB = (
    "P)h>@EdT%j2mk;8Apk1+X=(re|NsC0|NjpF6aZsyZf0p`Uutu2ZZ2+cc>w?r089b^000000NMZm000000Hu(<3c@fHh2!c|1eYjO"
    "vEmOFXTd>nad48_#2+k;Bt;PMDSTm1;0PtFfsc^fo9|rm+3eQaeM>yVOKIZ^r!uJ)yEL7sM5>c@wacr6b=ri=GB3Tsy}#slgLAi+"
    "$hZ7swO>Mr4mbh_a!`Q+T)+W%Py++*?9YG%DVT#1NO%|O8eAK1n_ffj&*#}&0?WQQBlPyc5KN=2iuK?)bM7Nuj{c@^7QMGbbFP{W"
    "N3ZAEaz-8bR-EC)87l6<^Ng}w<6myfE9f?K{5%DBLuVc3j=_L`@&;c}O9KQg000080000X0P$s1eE<Ld|NsC0{}2Ea0A^)jbairN"
    "Uu|V!ZZ2+cc>w?r00;vB000000FD6w000000GsR=>KhQr$WX>mt)7xvoLr=CrJ!z;W}&X5pq`drR8o|f7oT60k_r-cOUx-w1&SAE"
    "Bo?Fs`5G36I+{8PwF=||E^+pdH7ep4?FGT$qCEo?yeW-bq%_IU{?q1@Md#wB?C%KramRg8wYM(sbV#{6!QRSE>J%5pWcx5SU!%Dn"
    "=i4hBFv^(pa*O@+_sfrYn;*1)Ybi0m`QuFc?S_`plQ-|Qk9L|6VZLXF{q<{0a(quMgWH22Jb<hFE3`%5@pqEF^-|H0D95XCaV;ef"
    "cx0c(u##;;)(d;}=Ff%}@|W%VW`e*@412CI1o0|dy#STJ371c|l$yS{@FWg>2zUGz-eAx8JJ{ay=&O)5eh7D5FzV7-eBzOPbmy(9"
    "D`H>TI~|^De8&>EI|@EN5%bTxXn$u{=DOL>Zo%c7Dv!)??>T9IRPx=Nhv#<M|B!F^&-lZh$ne4hCUWhM3oYis_5TuIP|x^F-M&s_"
    "o%_S868rEidakd7jzIZG5dH+K3n$xNQUL%^O9KQg000080000X04WN>%m4rX|NsC0{}BKb0A^)jbairNUvpz&Y-KKPaCrd$5C8}R"
    "00000002n=00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlV+i=qoAIaUsO_*m=~X4l#&V(cT3DEP6dh=XCxM+0{I#ihB}%$3bhL4"
    "11<(I_+XDtfAM*fo*L|6U$A2Ht~4iZ`|IpH!LQ34>~B2WdiK7ilRaCZUzEc|Py1z}TqiUy_}WW}mt1NTse$XyJ-P5m#V$MhxetF$"
    "RAH~RKckdX$bGTFUWoTY0;`1xhJA$Sxvh_uGvt@qZ<91%#<i#tLw=XBiSjC~KKlbW<YDw2s5*3ULi%9z<Vl4m>(6D`bIEnwZ~T!9"
    "7q4miZ@R#Aqy74A^%_eKueVQZ|9a{^e-%!1Ov+9L7XF(D*9Q|9gqk~z_m|5C{!)9E8JiR&uBDJ1UNE{~g<?rmaR!`Uwd1yK@gYAr"
    "zes4CdSZ1GhQ6q~|5t^CmXd8RsWbpkO9KQg000080000X08|u7Bme*Z|NsC0{}KQc0CaL;X>MO;Wnpx6a%FQaZg6=401yBW&;S4c"
    "000242mk;80002(oq13bR~W#9N(HJ_lok;y-~}GFQalmaRjgGTK<ceXM+K|VigL6l9+g8{#TlnM*4C>i$Ys?jUWLFSMG!B*0})3B"
    "?+Zmhs>RV7k{8Wp*UfG=blH5r|FX$GNai=+d+&Sid%MAFqUX4&CX(+Y;g0?R!Lx%LUF{v+<_vc1WA8X8FeoHw=EAQ7gZu*&*GJ7<"
    "5FGIOdhpztivnK1b{hOyAN#@1J^R=%x95n3q=9TV35iQ7Tc>ocC?Q%~W!NTnK0(a*`=`j06E6_m+lAb8pK_1*ptvmlpge;}jJ~rh"
    "H25Upe7%>v+_f50-o3{yevQO*@{QYsTUiGuS#c#c`P43%XWMcjv2ygR_QP@s^50`(p6TF%oW*rmRLOxyF;9+Tn(r3LyZktR*2xov"
    "nB>^o?<Z+}uLTjR&oyvd{eIbnci)E?GAgvN_R>Y7?8@R!`HSW1|08c?!m-q;=ZMr>L3VK^WyGI-$25O3Z9hTRo@a*iy+M}82@enD"
    "edzi*Rl81~D}SfhW_Pa5k3G)%_d>MHi+vdKAXoi=)}Lwko~rgn%dOf!-(xF1s(OF39DOdqOS*4WeKF_1%MEQ0Cl=KDm)4voCWP<F"
    "D%f@qLmpna`)zn}$9lDgf4R8tIv;eGh>!5jm)YGkn1>xU3`xrxWQEgtSlP=Y`Toa}Y$gxK^j3YY0q~H{!^mYfQm%)!!nyukZm^Gs"
    "nQgOVUg<9|<l$PAr30cnUQ&I(J#Go7wuhVvE5V{$WCfI*dqU{zp-<Sd>;CO86A%yC(bVH1orjT!KKXs5a}_qXn{~JD7t67uYvv7%"
    "s;<Mx{jyj+{HEG_g75dHICy9X*n5~->bdT#ygTZC*&)aO<v&~N)bY^y?9fNgt~ST{@({*{+z^Y0E!NDo8P@J0VP0hFzPh|f=Xi+5"
    "hd%xu{X_d@izXh@dAL>Pe6(;}YrKE(_!TEIEN~_dcZKi!#UiejC{W#pu7ih~FnQS6^JVam8)EU0u3ttT-b<Tbx-#<>MxLh_n}^UZ"
    "^MSU9V|{JMZ%<Cg{N9&1Y)yYkcqMMReY1Zdwrc*iz@9zjjN`++)Ixc?y>;q&O8gRwoi*0av8L`Jny2KyQs9t{uFvTje$rEBf=^z)"
    "&}q^i^_b6@5$mf4e3(tw9*T#XRQI9ls9o?-3Gy(}qx6K!^?G97sZ6uqdYr{j{*de;Q~pqF9zr}c1oZv#kjL$%MZ2qs%9I<n#g9(I"
    "_z=d2288?gkj_JLzbtkSAs+Gp|LZBnmbU>9`9SyeWi(G=T3?2E2p+OW|Me8IhkUQ6fQQ^@>K+PhzRa|)*I4^)jAg#uRQB5-579h@"
    "&O_TFi8Y%G>qI|a7Mq7ciHExO%V<4?oW}+a!9(`wKR%?}_hl^kL-3Ff`1cUvA$Z6h;pY!w{~<TT?x9K2+TJCLn&W7mLgyjmZK&G="
    "52;|^FE1(fUs3gQHHPLXbp0}49tLlW3HbE9In+t9#8~n+booQR=PA%Hb3^R$5cWBMhy0`Mp<nDsssEW$wTF*0x4GPNzetex!8TU@"
    "5XHkqb{%7KR$FCjzWb(MuGM&7_kKBu;XXuTduY41BCP9!H~YfaVx1e|jP-k_Za-Q1Jyq=qzh9m`;Z<6%+vVzg4jT)7505sl(Awuf"
    "H&3C9hkYiMxlWPVu>O8%d`Rcv`LLw4*q$93J+xk;{G9TY@ICa)bs8Ic_9OhAKh8U-TAhCS>H_<0^qTCUY|!Q|J3`UEa^ZRClHi#s"
    "J#CKfHT9U3<JSz||3Yda+1C=M8y_;|ZMxkVW>+!e_551o{c_^I>9L!vYA_}b<1A_&MpT;N&@aCY_xvG>hjjfi>_4P#55|X7@PD2I"
    "cnBV{hx@u7@{q1yhIKva_P|3b==UEkm=KeDQC@|0&0Cr3HZw$PzHDrDqu`+-(LaBv>wFnJR01ASq5pmxvWH@=3ow@b*h29T<!$6X"
    "YXcMR-y%{&EvM!6u)yiggJs(10C^itAa6qjtS>{qtU!1kvYn?G%ef-pp&@~~0Ps+O@H~WgNDX*sNT5CiJX9b&5A}_QkjDlO*~2{^"
    "((Q9V9>V%Eb$j3;6%_jq<8s2ET3Fz}{=2Rs((jTwe@OQnAG-ZEkUyku5AugpK)oKsLj}U~km<Q*yS_N$5FKTP>lzQiLruU#D%khS"
    "upb-xW&Y9ja76y3kT%z@5$JgVS;TO!Yg<ax@sNC;15-Ri^JT8}DNJ=z-D9j;4DV#gSf4UO^*QA$!_}u;j&d@yNkerY?Bn6$>B`?p"
    "-};oL>%aV=ZiRU^dQGlRiCD9&r1fr8HzvIL6n)R(g!L5Ym)Qg3L-0_6@H~WbIH>^-4T<odM-LuyL#!UcIql$~;edKQs0&aaTn`UP"
    "Tf0v2u*AC^KIHAS^Pbi|2UuTLGBByu#(Szc?%QhYep%pqIzQKJmg+uq9X!-TX!Uwf7r+g%dkD`901x?x|ML%Fz6>6+NBI3Rc*qU0"
    "dZ_F30?@i1j1Q^X0}rX-|GY2o5IkfL_xcp%A>BL$>h-AG0}rX-9uLucnQnbqZ23dTV;d6w>qd<&j}0F30o12JJXFwhJp8|&CnXdQ"
    "As!kM_UFzqtuKqUK7}bBLVb!pz&Y*EFDnq9hfuFa4b<xy65Z=l$j_6~HGe3UcnIS|L&AN18O>AZ;vt*|OWhuLNCm8?fQJf%=OI(S"
    "Y^?K?#;&6`b`OQN&!MS#2=R~$!hc=>c*qU#5Ih7A*+TE(=;eKP+ze`i=cd1ma`nASkk8?Sx&S3mug3@U^OT`QDF=!ZEb-bS(_|xG"
    "-POt;>iV2V@K6)*kP7a3Z17MC^vis}eV+rJhvfM(tm|n4>v~kcx*o(s1x?38x_u7t93Svdf7p9S9v?D2*9<(=1U%H_KTt~p1T6pn"
    "00;m803iUdTa8@*|NsC0|Ns9C02BamUv^<^b!BrdZg6=401yD61poj50001w0ssI20001;lif>{K@^84u!tanpo<`4BVx-|Fd>ps"
    "dM~;Ow3H02q%wCGv(RPLg>tjGw6NF`#gf!MHPUrm*0n6{EzQ)*ugr|J_A0|5qAn9j-{<@V&CSE_H)qbAnK@H+;Bb1zk!8xK%Ir>8"
    "ah}({+h$L3C)*u1ySvC+;>|5QQRH>Hmi#PlscX?L&d>F@7UkCDB!?|&o5Oa&_TOWbS{gMGa|wFgD~l?%10DAUef*w9vuo6_^okxB"
    "Ze;#s4SFhszSNG6XPwbHj_>Os@4{R3Ks9mNiSL`hk32-r4xx91iJQKU?-;}15W%+(;O`w_zszaoJy)1dtiVqgv$Rq1nfN`giOX(="
    "@lRkvJN(Ugbju6!{OyGYJK@|UanBx0o=W_)qSr(ukL35<WIpMmtc${cFe>L>J!Hk{OcVddFpPOwUvdrYko(-v`Bmk0)-}le`eog_"
    "4%Tgs!ro!H(968K0T-n1*)I6~4*7TWqs>~FLZ9eA{p2Z?{oTZ?^a1=-$=7%aofbs@l6+rzM;#~Y%j7)+6>y>ne}uaAjPygW4KD8&"
    "pZ&~&QT&+n(RTJR>UnoH{s4cX4(2!D&vRb=Q|ihN!}FbRQtGmK(BWp$=h%PCpyZXl2+6&!=l|#e`pp=8j~}^BT%sSw5wCsHPZ#AM"
    ")pJhW@ImtNue4j<SJk15=zI008y=H=zO^wQ{e;dztG>IintL}^QlHus!Z*udS_>=;!i`U<C+8*cS#p0zThXTcALlLjNdA$hmblZ>"
    "&xhokYx&P=pVSrdqerE$XQ$AYo0zYXa~NfCo_8?)k}q~fNS&&sn{@&1Nma`E@}4uFan;gB`3!ol(b9USyobNf_y<r+0|YGq000O8"
    "001EXI<Ylo|NsC0|NsC05dahbXKZg`VQgPvV{~b6ZZ2+cc>w?r0Ehqp0000007U=*000000GsR=>KhQr$WX>mt)7xvoLr=CrJ!z;"
    "X`!y8pq`drR8o|f7oT60k_r-cOUx-w1&SAEBo?Fs`5Kx!3bhIp0WJn8002-+0|YGq000O8001EXFp5MD|NsC0|NsC04geGYVPkY@"
    "Z*E^{WOFWVaCrd$5CD(>00000002+`00000005ip7wQ`j$;eQ~P_3SlTAW;@Zl$1ZlWC!@qoAIaUsO_*m=~X4l#&V(cT3DEP6dh="
    "XCxM+0{I$7I+{8PwF(pft^=*pSs8%f(XUBh8URpB0|YGq000O8001EXu@p&?|NsC0|NsC05dahbZDn*}WMOn+Uutu2ZZ2+cc>w?r"
    "06-A{000000Q&*}000000KJ#bYTG~%hqu>0h3UmDm}1G2Wrv=7Yw0D2Qi?IQ(i+k@*lt<~d5XSpS9Ct~i`ZF*9BLROtv=7}&cD_E"
    "`FQ!^)8`lAF5Hi|?QwG$UyR1@E{kbXjwhq>)&6igtZ!EP!?x{Sy<cA++c<vwzP@ea^XoU0(ZlHZznAwRgjZ3nqO?)gQBF~QMmgBn"
    "D$3EuyXRYb?*4ZXWfbL0lp@M9N*U#Al!?9TxF1n=QEp6MXJ=zNyYaoe`!!$>GOu=EcN*obUDe)6{)aBE-%RgD?4tX(-LGV=J6mtZ"
    "Zp9}%drtYH^TRAkW%rfIY%td$SIcvbo133IX);`6aVc%w;}&Lb_5_Zxd4_{69j-QeQ=4OH^YpmF^c0R?;7Ykq`Gt4t<J|m+S#gG|"
    "%{KQQH@EBZ2^=<4@lXz5=kztk5ND5D7>izyBM0k@Ux*bs?Zu-o$<+rOd0d>)Hw%4Ed3l9R=9$x{dF{ocuq!p<#e=vKlkyC=5WODP"
    "h|b*n&_)07IPF=UUGjY<V;-lm%jtuyGV?<Z_37mc9Bc9XQp!5dadqhXm6tE{JZd<%jwU>=l@!OE&{t)+YOp4c<C^QcRp|TAIlJ1!"
    "y*wE97|$-buKhQ~X$|G_uL*rU=K97m^!<YVHst8vbMbd7Zqyfj#j7{`OCQ45DUNvPo}S|HpXN=9YeK(gRvC^Sg`d6k4t<p3uu1ND"
    "@nwzbm(+Tj@VI`-pA?6m=fUr5jm$}VNQ%=QO>KG{`Q-I`?ca^)OmXOBpGnoF)&<{BUOX0ZMQ4iRo_)ldAKxRow+=b|-qbwwIQCxh"
    "FvSrI`gCf3_`eK&y?gpn{XE6#-a-APIOdk?yTpP#TxR^Y6xPcF)`)HI`(@Za=z+s?++Fy0rn(OO`G>RF@ErHQ&R66JvGmR_<cNNg"
    "<~Z`rJ6~yE^!7(M&2#VEL=M%#kCXN}JzIJl`wMgQIQpoUzi{aBzDJeQ?;DRJzZuJMTK`>MK1BHoP)h*<Edwn8000O8001EXD*9<@"
    "+5i9mOacG^4*&oF0000000000fB^si003ieZf0p`Uutu2ZZ2+cc~DCM0xbhA000080000X0P$s1e2xJC00;vB01yBG0000000000"
    "0Du7-0RRAIWnpx6a%Ep_WnpeEZg6=}O928c11$gm00;m803iS=3c|}t0RR990{{RK00000000000001h0qX()0A^)jbairNUvpz&"
    "Y-KKPaCuNm0Rk-pEdT%j2mk;8ApleqNh7WZ000os000sI0000000000004jicLV?cbaG*7ZeM0)VRUtJWpgfWaCuNm0Rk-pEdT%j"
    "2mk;8Apo#jja-ic005u`000aC0000000000004jiX$=4XabI>}Y;|RGE^csnP)h*<Edwn8000O8001EXI<YloMF0Q*hyVZp5dZ)H"
    "0000000000fB`lT003ufZ(?C=Utwc(X>V>WZg6=}O928c11$gm00;m803iS{ibM`j0001x0000E00000000000001h0lp9b0AXWv"
    "X>V>{X=HOQZg6=}O928c11$gm00;m803iUe6iJf%0ssI&5dZ)Y00000000000001h0ZtJB0BvP-VPs)+VP9%<Z*DGbaCuNm1qJ{B"
    "000O82mtN@004{>00000"
)

_TRACE_BLOB = (
    "c-rlKO^;hilH`BUIS<LtWbRE};&chEDhZNmh@J+~Xkccrz@qoiySKso_th$s<l_r-b2E=%N?p*6Vp8wLcetCIoB!9V$JJ)B+iaJw"
    "{>NYb>(zh$?H_;tkAHmm`05Y8y?yuQuOD82`uy9+{j1~MtN;18|NZa(=gVKdeEj?0{_(&5<Ntp7{Fhfhzy11G?ZY2_`}1F4|N8cq"
    "*Y954z540>;nm%;`TEn(`~8Q>A3p8(Z@zqe_<8^O^Q*gu+1J19-@X6!)!pL!^A8{2zxnN_&zJwPJs$u2ZrZUAZ-4sJZyzpiT1@)("
    "msf}Vr_W#8`s@3TpMU=P>HO8~!}UOX+P`~udFyKa*3;uAuL2F3zV`BIJ{70|v)6^Q2YYyK$=5kqT=n(&SL9tE-o5^5AB`vK&*?va"
    "x6Rs3-unDsrsLVP<Lh_7UKYbuU!P|x_*pu_``7dLUzW$~&-;(_MKu5JdiB7iyPPkgPw#)5FQRsF{>%U08CT!TddH@+9h~a{o{iF_"
    "e?Pr`UYe&LJ?+d%*R8p{3|ITo(<n@Tl};Dfe`s>RPH0vzdCT+IgE5;7M>AvXZ}b^^9(Ou)SI?dAyvq=_(-f@Bg>blm%@7`~{A{`E"
    "f;O_~(8(w7)KYyc<!|zN1Vi|I!hku-=1m{O^E=KTzMQ?E&<AheF5}+w;OAe`Ngw-sI^jb)aQeTKHw}Gm`r#Wqc6O_*0&6lkOpOa<"
    "%v0xQtFwJyyajW6g#5HIBgVAgr}yvP?SK0G*MHf6{QUOa+kd-069x@l`6b2@DZk@Jb8xz~C+!LM(Ag20{W!SFH%G$){I1u(F~9RZ"
    "uIt`yYX5cGB*46D%*Tll4i;|3&j7{<+!MH0&r91flX)Mez0LZV4j^#s4TF@qD)3YGKsFZWQ~E&W5s3C<hd&xOx#&Q}gDTm+$_Apo"
    "c|QNd)9G`472qj-9Q2kA=K&b^mq)h7V7~cV;Dp$gdHbx#rKT#u&0g5B{_(W&Pm}L`U_-4jfW7E43J{Pini}k4>$~C@niJg4sdZ4e"
    "1~WsT6RMLgh9L$5!WqL$qj)zmaQ$*z_l*j8ne%9BEP%H}^Pk=bnhjAhPss3at1W*t1*id>W&wZ%=b|Gr<bZ~&bm+-H$I^a3$k~s_"
    "{&=j3NyXNS9jgby^dKsqSX!IPnOX70O>m^>@)=-AU-U4uyDA2X2Prx2Cn51(FO=Qs!Pxx#`s2Sw9qR>*Cf(6peK7=$hU(mx;t);8"
    "qK6;Q4o(|DJTn2hP!K-o=Z-zS)pTZnJ#rqD=}6^R1^`E{mfdBIo(AO-u9OG;@<ep0Oy4)RudL|siC{r*F7Sp*c({FV%W?W)ApCf2"
    "FT%5NY^2TCJ%2^x+v%WBXhc1mN}oP|e0})C{^Q3#19qSkH)5Aq*l_Un#jytp$zx2{Eg2GAI_O8TZ$#<wvvk}H!*C8?^+QTV6oUnw"
    "4!1IzruN1ZqVg~!9?a@mALkF}OFDm?2E(52*vU4;T-1S)$1Z<kJ_5y65cS*o`l*>!8xMVYYG@_m+1^`(e<o0;oyS#hzMHY*e#Du+"
    "Rn%&0+cYm^sl@0(yVIUu^<^V|e2>y4OrTxykE#2e;VYGUg#i$gTk!Dy{kyMeD$s~}{^RoueK{cC2$8^xFK&hF%JiW-IH?V{jFA%t"
    "u|?*C8eIUg5qp{(dB|(f88CD}*3al$esm0sB)(KGTL~5F3ZUwaM<3Iujr*p+l?0|XpUUv#z7SRfO#~?B<x>T$L!jS;e*`SX?yX>q"
    "Ks)E?M4O)tmA$~m8htWH2An@FvOAAG*LYOGwU}9{w=#ydl^j$%Xa*4%7&cqy#udyFwJomOP>M2}>9SlJ%Fekr-eG_;Jl)7xjaS*u"
    "3<v?G=49}7X0CCVKs`Z#cb=#H-jy?=<#b(=i<r?<W;w5(rHJx?>3ZjBy)CAYj{iC61Yg_Q3@W;4UD=m#okfx|jnMYJ?dc#6d0#}`"
    "vZ`_G*mbVkgyF3kG{m@k!Cq^3I=S9Ki)Mc-CbF`tKmAbh13`boP8B;$=D5;L-*zL@Jw4lyMNX-77>&&x4Q5N8-;)B9UYNffgY)4!"
    "Pfg(O)J;*)&5H<v9M+zGc=Ps;+N;5Om!qe}tPNm2^u01q{qG177dr`<7MfGh&#YVAV;`YGMo%Z(FK^%d>DvE+-VwTS;Lli4`TVXy"
    "?$G^F-@fSLKmSq3Z$$ec@Qfe@=e*l<yT74CDa=O>7uA6S9rdobfBMtYL!B*kus(wKWs8M)r*}LXPuwyp4FeG)TH_o@?jV!AHLtVo"
    "6X_+{#iS%<bzy<8fPq%Rt~Dk_dud-dlhj7XT)Z+UM?6Lrs3)-|E{!F^_oZcf`PBA-JJy&`S($6*a-t@aF-9=2IlF{+34%$1HLf7|"
    "l+ICAM&g8SwuM1k!1gi+v%Tfc&jf=>kCR}eTYj_QkrUX(^h;3`r2TytAT!-?in|KzvTmQ2G>@HIrc(`PQ}_Hq>N$Q1Yhdp7-ex=j"
    "mj}F<ILM?cxn(!Xw+o15-v1cJsA|O~1$^c5>-NzV%9q&+9e&}gCFHyc^f{>|Js+|*E{?-PjMoI{HHkXv-n1bnJAE(G;UmA(l%&e7"
    "ZgH#D7}ULd!rS%Yp*gMF_w+$iv@|=a4FTP-m4sGcFQq8#P!VbB&-Vm~Yin_UP=^-htUw%|lIet*w#MrV*mT{|W$Iz;NwG6Z-1?}y"
    "TTRy!x*QACn`%&5tI3JPBsBycni@H;1QEe`)UFpgczY@4sPmQ9M~BK4$5N}ur00&Ydx+R)lCF2QI(h<&lG8}Pk{ojbY`&9BD&oyC"
    "?_Bbca&M_yb=hm0?lKh<v_K0(DxKIcLqLVe3u4@Ac6NGE!pJ?SDZ+6Lqw(=Y6*yM{x{&{R;~1<L^@WucZXL5d><#@tYj^Y{?R=lD"
    "0tPgBT+5*N9tNxBSd2vngd9K_ASgS^Ii;sNv_G(IU9tMQfSL?4oR`eh^DTtxnNL<RnjemRV#P{#<Uzj)xm1tR>&`3J1sq5frNMXN"
    "X2C!T4B+?54V+(@y@`g@Nszsw6-r{%)a^9DmKMRPO)MRt=}WS}{vc^;_0g;;FnXs{E}O!<k1n$V>L-N@3ZJS7{30tcz~2jzSGZxh"
    "iw{}|{&U*gMafmB9zH$_NXlRWoLx0{*M>JXNBtIvlLDia!-CwGzrplxxd+>rl({gB3)W`(6&)3MCBcpv>58$sTEa*;v-GB=7>S%6"
    "rprzLbG3Vvcn({+mnK8G>k($w(YLss<zkbsNrtM<24WjbZ`@^?bDVc@`rhOu(XwlW7T#%kL}&%qW^_?<n#l5kO{`Ogn||byr<93`"
    "O*cOJ#I7OQh^i~)X~VX7WiZfMGGxAV)*^CFh%%uR<4{QCy4kyKvc!xR9b{rZ>6{EoH(0I;?1^oNrnG!<IgMg+^xc9(kyi}19POL4"
    "DJ}q`6reAv+T<a|B4Hf3kpdzk9rsxxcvIY=80_pV>rv!^r1wP|HPQ=1D+^A|!(oxFv=fCA(k;oDH_O_!p75S<>F$mdG$_UIfQdl3"
    "k~}D1;R3YFg}DM+yg0OWY<eOyiUE&EV=<_FhT{4ukoBoLL~((zu?S_VlIbAJhq{tdg9SuPUPwzr!wf8ZiFKdI+3z_OUps}8`CuHu"
    "rbc~8(-#FOxM+WMNfGCZL4aU}_Cm@8onj(?sNM;L7sw(7d8I6V5DZvbirWAb{<&^~4mQO#CVFN4X+?#06{s!eQf;MT3gy=%mo`_B"
    "E3Ox%JZj|()E5e0MaJR*E$0b;R9uA5PW-cK14Xpm69yS$`V1GfkhpWgY0-h>^owGQglEoQPTR9f(40Zk7~*OJkcN4qc`|&>14fxF"
    "4m(DMSWIq1lCn}GzmTv-8*HqJ3;8)pLlSK$P*ihv!OGBOR?w$%y*1*P97ts)y@V(+<pm4(@mQffQCNm7)cKDD#dkxWLFZ?2bfcor"
    "z$7(kv`LdY0X&GB)P|J#UX+r9Gq~*Hu%z>saYptXX*5WUgp%iA6F>%Y^NmjVSReJd=3WNM8DJi(8Xw=`U|=mO3xlqidn|3Uv>y+K"
    "?@zs34F-S&J-?AyXPA3o3<i#w6Y;u^-Au$TXn+aeIp<TZrBo8vS!Ru`#9%k84>7{mXQ3H!q?t=!XA^~B>EIT2LQ~SG(Syo><N`8P"
    "9afT~LQ9nbY=I$~go44_XFKpu3DH-~Eb7uP9)j|S-;tZYDbX^fl)Ia{{)eUnFD4wPcZrY%qhJ6RpY;5vV!&Vaek=KWQqn~x!j}+^"
    "3t0qUep`*ZF|rz&L(8cMok&r;N;H^Cv?O&-1l7h**b&(X4oZ1R_YYVr5LnMO@oOj#u&mp^WuTJ*uoN<kDQi^n!D%?LumYKspeSOQ"
    "_-7qCOw3o(5^(ZKEWjshqKMN(vk(StQhuN)<z<USktF|Jg&X!t)>KE_UN6(E+TNxps4$*XmXvxB%;yDpMP3;*F*vCeWMDD+(2vVQ"
    "8UQh9@d_nCss`puQih$8P*okWpjXL;gy=cI*&I^ox1BZXL*C}^B&t2WMBJ0#``Eg-tH>4pc5W|EK0pD6f&MEHYzFo08)SQo)ET%c"
    "VdNURcV?LMFgz4JAS|lPP`BK}K^-x_Lar2PwyU05hP<vOp(}p!tgn9FJdGE(d23ha)cw=gK7Z2{gH&kpy5C3+3%%wOt9#vm7g7{g"
    "aC8zK1Z3@IbxGP5Wm9IT*g<9n6*e)7Zn<Rv)bMbn=7hFk;tYE!8?5ax_2B3j7~U*!9PygMfqY2_bl>4I&<IE_hidg4EuHKAVsyOv"
    "W@U_{k+9x5AWEBKOP9gZsn^OFlQeE40*GEO=HBJu>RLvL2qZ``>wgwfE>P8!y%yjlWNr(zNyfI{4ba6#ixT~Ca;6z&j|FHF3l&Gk"
    "sES0;=c00LYmK!=kQ_=(1nmovhju|+d3JJ;G*S+$M*S+n_+Y(^&GD{2E<U5FdwCZh37U^3txMEQ#m54Vy0%I3Io8QkmAc~9rH5`h"
    "tT0yAnsex`?QwzT2?O{_VA_}nL54}@SMW~oFjdH?6lkYC^#)acCz%3&;CIG)@#-V!Dvt%$+r3km6>l*btJ1X~{WE865l*SwJ#*EP"
    "1w*prJ7vfsjy1~h6%P)DoFwg<Qcvn6TSBhS2$&FThdaJ+-f};gWf|m&?J|`q%%m^P*+Y5G4s@_tHa8gz9PW-UouXc7HQ7O#PkKG^"
    "lZ7sYjt)PRJ4~5jl0+)qyorl?^X_q_Vm3j@O2S4TAZhn?R!<(GD+?{pH*f!#DSD9FjyIjng;U7UE9uA&nuwn;vU96JqJ~$m4BIkQ"
    "CPW6w_Io84)T&7I(B%0gM~;sg2QTx|+6Jb8^uS<04Qnx$geK!y0LO$F)L4DMO31$hE@hH>a*4FZqf>G$b|Lh%-W+)96pNJdG1N5j"
    "P9aAzCx8}_u+|gEQJ3%}y~3hCBck*ms;oO*BFaFJxxj`O2eN8(13^iQDV;^4s(L;74q`|vLMJA%gcNC8j*3N<_$1L}3w;8i$P49!"
    "Xfk&twK1lTW)zpqoUF14Afa2Z>8QkCib|vmDhnWZtpM#9Qi|d|(FByel1@=dW#-vHN}@A~l`TYv+)QQH6ap*kaZa!Rl6_u9O!2r}"
    "9LKXSroP@ffNZ~wt5;`aJT0=-&6A}8IXL4l90Yf;L$01_r_sAoSAx$-^u3(*)R8`DAU^r)Z5Zq$oKA=X_p=+wE?Ev*UD@ppj!U=9"
    "LxFs;(a9Gq)=5$BYfr@>;|l`LL!{I|<^s?+PSek2?Ye`cc{qZ-&;lkiWPzN0={n_A-_!}R3Y;AzE0iV&TX`29l2k+l0W^jUp+PBu"
    "chAb14uQ$TvqjOToG=ib5lwe}sW_%4FBP0k$?@<~s#G>CsRr@{kbLO*Die7~8>fyRBb-U%5P7H6i2nLGOb?M}holrc{n%d_MIb~u"
    "zI~d(ZzR%dD;s&Pl(A^mZ()C%ET19A32<hSfVZB)(<gkkG78YODdXliw@<R`{Ys_X;s#3#n)Q>VGW9oD&<GG)EM>cC4JJx>Uc4V!"
    "Q*?yIb9q!ee|aB3Ig8~t%cS8UdNQR?U0UBfg(iyzSxGd4>|Iw14cJ48G2M~Qp{pR6sR{<rRKcVUMG=KT6oPTug9pt5(GU$O<))J#"
    "V#Rednx>w!Feb{+?3HQ}11aG}N>z;_;@quCM^#B+$-S&JgeD}H7Y`Y*fRR?MkPgK^Q4hV4oWi5nw!a1Z7f+Lbk7KX`bcpbf%bb>@"
    "@s#!hTv$DK#?fM=TANhGLSpkXZ9<3{X?Fu0SpdE;j5^!%g0!O(m|(0l0<;9e3mlqcW>pr~o(E8^HLa-5*a9Yu4<e{euS+w*5>g0L"
    "g|=ENr4X?R;K*zFXPNA$B;7-K5vK_MT~+Wwqe3O&4K4~V!H?0R;mH;J>SZVlDg+yeLP|s59G;6s4J1R%Fan!^%fjo(HLjN@h76ZP"
    "Bm#}|KBX%68dKSZO{N&haycEOuLt;gFlR(ZO*9HTBJz}*+nTFUiX`PGCq!OHr=x(FBn%R@dI6`66sZ<Ew?y}*afimzMJ(#YQm}ax"
    "nmk-jol=#FN3Bz3CdpY)Y}I61gQOUP6Uz}nWU=VUDr%Q4-bp69C^?s@A2)_hQ_81oc_?bF2Qf=u$!bn#Or}Bs$})#2sO_1$-St%F"
    "Rc*S|iJ%+=JriA54as{$XgA5W(5ib%^ry5Zk0h~GkwJoLf1m>qE}7Jiq-ha;Fi*<DW_l1{^wr5kGOU~kGp0K%HAO1=9>6l38WGx="
    "gXV~&YJiM3C2pWqj&v9sHxlVca^-UQPOT<p-n|E5qG&pXnDn&Ip+hzlV^$m^?`ev)rj&!2S~ajgAg&}OGMz#<Q;&Ts6hY~os^)#Q"
    "twxt9*lb)Eb}wEKeig}G?{<S{dnl8j2;?wfwlN(do;b==5GxpS%wf>fXsb`kg(gR!Y_l^Id61=1V@9p+F}SDg@<ksIi11;JD%JP{"
    "cJAAVygkAa7IUvlRUj4aRbQf<TGcd7<CANE7>UAhRM3R<WKc;jHgfRAi#?qVg8*hI!j_)Ut1$u6LIZCn)#NP<J4<;FL^TSM50vd)"
    ";Y?7HK5Gf|QgXsl&+<)AF?h^mk!GzEy^9kVp;>VkV^k~hM<{(B*#1i;{KS+RC}5Wd3k@yh0F+^Q<EacSO<A!Pg`UU>i<e1NU4b{Q"
    "SK4SVLIS=>lpQjzNv_oh#Ss+P4|$2Ac8gNBL)wp!6azUJp=A~wa_Qlhf|vT5p(sD6cpU^KXGZz4A+A;3uamSqNk6ing|cJf>=@1m"
    "(s~Hjqa`Y@!^u&<gzl&k!5&VYO$396qRzCdK??giAwh=a<aXjjR`yU-f=}b~Wj$xmSb><q6Uds;O4eal*jHr~f(h12$&1zHV-a8{"
    "HMlMY+(i^yH&_yA%Cn@xILw8|UmTuT9~x9D?Pt-KV%q{JcyJ(f0I=niZ@Tj;9YzX|Xav@#w^t-f3DZ`rD_BTFO2*vaNu?E+g??A+"
    "n)c;fx++jrVP1BZY>jPmaG93s?vgYZ=GK_N-A16WOht$RwCHmZPIn^z51lj<1B|-NO{4gqr3>iAsSJ)#TT+ZHNh<RJ1PGq(u`dhb"
    "<9lw9iJEftU{|&{dDc=wD@>dVe+m~%a}KX<wdVnEjWvJ~`dlf5A+CbMDOubll2P;CI<pUVmFxAW369ezS~R5L`?p2MCShBxI<~4P"
    "rU4gC(0IUhK4=L~mbb}dZ@IcukzpIo^4bwql+rQCpWNi~;Uo(bAxgr0g*}@I0jO+}ZQp*Cmh$vYo$p{V)g~F2ECV;(uA44OvQ+h4"
    "OFE&sQXQ!aRTeU0W;GY*P8U?uXcN{Uw9so4Ifs*0%3eWW7~MAoM{8_)C;p*n4utVWJ!oA4D3@%;%Cu%K>X15{nV~`T_RGl~eIVt="
    "1Np{W;yl5%o6Q!{^q#AkZElFTGP05dZYCkrha$NRDEMS@%&Z!aA>NeIt7dPsNH#57qA0h1=zuiG<ohydVX1<bB&n&A{H$4EO7B1)"
    "3XqC4Eue~~lDtAdCvxp_t3;Pq#4l;3ly;Iv<5;F*MJR|G7LC2%@>zHpG<G=?vzY66SjW5-gyOB-TP!qd%>%}mXrQT#xLGjX64q%C"
    "I!G8x_>nXll_^XUmQbopK<J2BO#yMVwemzFqrXa^;VdyTb49C9AQhgx>#pLZ7IDNAb|#*nGEHeo_91ct$$UXOvS}F%2e8yYG8U4!"
    "*1C(ZHphlo!91|k+F4VMhe>=V6*U3!Nm~|U)c8eQ`c3ms!`{Ti%67QKkd=`Bc!}X?>w%RG*;@uMJ$}I`sRb+2jyjbf(7|eMOAlKT"
    "H0Cz?LJm8fO6iHhHT0%eQo23xkOTad;$&HR@l1pZBD?3m>vdI%+6Gwzq}Yq10w)()ap2ss6u;G)@h8Csx<zhfmUT#u?Gf7fT%mD@"
    "teE9(f$dbl$rSEVrf*j)27h(xCuI|rlr%2bH)MizuuL1pO1h%Xl%p<VMwsIQhLVvWuo{dM+k~B=A!<fMhFG-sM+1^b>QU&~;VDbA"
    "P`9f>ypd#~bSIM+l=G}nAzZ`OAVz#5@6pM@l*EEOMp^S_pKZg+B65>c7nbxVwM9wG#026`sl^xNTToaL@;t9Q<~iftXhcw6h=HYH"
    ">XZA}M^Io3n(GyEgf|#2LP;CLJ|T<sw6-R8?D=mkevut$oO@zyPKz}4=$yn+79v3%+1_RMZiS#cnTL*3H_^pZ03Ts<_F(`roVTcu"
    "shE`<o~Q6GMZbQ!Ji`4^CoC_PePps|b<^v!9V;P=RbK_`PJW!UZYkQO#B0k2dw(cJt4vN)!O~ilL&$DSr!z^MW|0OZlfIgV*)ZNZ"
    "@_00XKz{b}-vwt)ilT+aEK5w57qd|2GoeoKUW-A}0i4IN*yZ*0MrqJ%B-TH+-ey?-bK_A1H&V6G-5}EAy<f{?fdzb+F=Oc5(e!46"
    "pqZpw>Ip@FCFisI0FOln$hvD*(#uvaB%7$Q=JZH~T;5$<gi-kP0oHW^jj@%vaYXki{2y4(DJH0C#pxVY94O)|<jN}n0z+a?(wk2g"
    "XrhkQUEmRnV-^IHioQ}yTcO~!`$=Z6<zuGR+>uye%iJB;$hp+O=P<Z3o=oVRNf?OQXQCEiI(p<7jiZr50iGzgCMBhm!!t0?9m5UW"
    "v!Hd7T}IKS1NLQXjCL&<v81t+vvZ3<wJR(=mPLiP1O~n-ZX$t|jJlP}W5JotF|U#3GvKfb;*K<47~iCl(^=1&!*M`K_fe#zrb>bW"
    "e{Jp2Dn)86jpY()O+6O9?ocm_RRlBHo!02%(-ursMc)6RW2dzUUn1;8qgZOxPtB!Y)j5Tuh3Iwx(gWggSD^lh+PS)63Z<Jubz?T0"
    "RW8%|u|Q@MRqNCWX#&j%Zg^SVze;IaRMYzKLFNHGsRgB!voetvoxsKkXCr6E5r8KA$SS(VjZv*)j-^hk5hbYg#Z#bIM`~RAM+?MF"
    "0U4=^8S@8f$sm<;33!yTt>KhIb`%3iD0S(qu7IjZBW{$7mKJ@ub?H!QNJ(@sy#1xjjT93oXo#e1w2?SO)0e8F2x>4zQMGhAx^O_k"
    "FXp~np&V?d^pqqAc-XroUs)I?&Veq9hb@h?1X^59l2%pn&Tg*#)~?gjM!P2a?O^gLk{+den}U30OL*&07;jf8OZ4;p){4hZ5*rU%"
    "y^WjC_tSK7#tO&o)w0dk%+Sg_Z@M#ER|ZC-hYs>o8Y*G?Qd=9qGc5_R*}n7b-$=Qk#jSCE>rBVp(DYD381tBCoQuJ5%v+=#JgRxY"
    "-Wk9FI+nm6D`obH7GKYJr(z(j6fQ{@+$3>!txjeiRJoE#X%Lr;z~g~TwKl|&ir77bKhNRJr*8Ih)vPpwmb7GHpLvm`5<Ly));4ia"
    "Jz)@?DHKris8lrtI=v2SQNqqhvz4^yjO;gkLNr!iFSSh)DOC&Vm<TWl3kL;Yoq{o59YBwD*AUgX?vQ|n5X0TW&@)jR&Lm|{K8rOQ"
    "Co6v&kc^{HC|W3;s)#D!9srk7WngQCLHS5LjV$9*5Zkt9ksky!YgiKNmKGOFsSg2viJ&D`|0uFYGs<|o0C~oIPlfa};SATfRt{MP"
    "-a6}k5SL41tT^`pN(IvnD=KbIsTRCGLsdeJ_}D4oh|{ig@rgbbk2%OmmY1?-gmRkeI$1}n#7gzrq`etQO?D)XX#oN;uBQooO3wI`"
    "nAQ|)I>~FHV%v=>7x_R?34e|EU<NxGG5`^!-C!b<8XT32>ox6vS)r~ZBuGT!ZPlGcml&5%jj@FJm4xII|7?W*mRdwq%W&zIpx5)b"
    "9Kumu7tKhzVhowYaaC%!;9I|+4VF<vbgo(w3+ZEhQPz1@oRlN^YQW}9nktHqNgi1iaWr*eR}PPhj;B+ovl`#m3%qjgf-7HeDux!Q"
    "H>2-L0+gr<ly%f3D>}7l84kvmOrl)?es_w-R``~Tkpq;Rl02j=c}{YJGak90T*ik~QgK#srjjCcDR+YVIZeo33rt%)gzlWHMB{^a"
    "zp94gh!bk*bmUB)P_RgF&;mjHam$H18%!aG!KfBdLaOR%>})A$&sd>(rI*6UuPS&Ey>;mh(90QG!_B}TMNTOdA;P_waEo+8j3iwo"
    "Y;BO-&XyQ0HmGXP3W<{6LrU6&;wf@*c7s_Sajn8N*58=^RnZ%NiGqV*Z%Vl}mN!kHm_+QEz(`K&ydbBYQZbq;b-Iu*;R%ed@+YZ!"
    "^0p5eQKi&hJDnO*ZiPRyxrMPjxE^!O2Ym>ah6+GG<A)*`{67uPQ7jZICn<nMFe)ubko%O;=e{zO^kT>7rCO0Ww{jKaz)%9o{wz*w"
    "rIk8znB}99o5^Y<vS1UeLdz~v!{AB#(%MTrawYEuAE9uneQ8An)wK3J;nSPu{2)>D%_N!P@>FX@K;`t#$}swzoD3FL3TyTrJ6iRI"
    "Ys4=uaHXoF)Yf8)Y{jvB!T_i=jIa*ya7i?So42jTgNQzK9eEb%;W<$NKoQ$bi5qd1P2sw$@br;9LDOIEle1-t2abuC9=5qO1aXKf"
    ";cdk;z00EDiUE#Ub(+SfM2NMzofW;3%YYPo_bf7))woQ8gBF_+Uqh;{d+rss2Po<h)k6$hsjRy8{jdykOLGHX&Hre<Ho5P?#T7Ty"
    "dNF$iyj59CYj1|H0fV4ijy)Ij1g5<%8`}9wMvjasl@5C)=#)xXO5aOhDAH4^Em6rK62&T5Mcm43S&c8PGUa#M**72Gf0*UgKYgQp"
    "Kz)7sBg;+QQuFd;?CwJO-U?fpoGop)a#MVlHR5{Azx?ZQwOeeLo9&l>PygQ^etY}w&0oL#@6W$ogMQU8&~q5>^{;P#dHvNYe}9;0"
    "7M;KT^z(lIVe*ITu-!bG_sjm>`(IBXG>BYp!$9Bjn-<va$Q9NtLL8DwXd8kT?tCf)n=YI^T!ZV>lCN{JFirMzh#HM2>d)yvfVa)s"
    "ZJkSPJ76^<RKP?Dm>Z6Y!WyguU1*3sj-V1kQW6ATB)T3TT6X%;)6VQK)1Z;BDixN|<ba*vxN)FfPCFbSN9*z%ea2Ww20nMb^YUbu"
    "ma^H&E;8ct^Dby3D=O_e_+var$~XBuGSVv-Oc7k%co2!TtQ-Ge;E+ns5Z|d8;<3-C6F$`WV?c1vb8qCUc0G1>tE@tElpdzW1v2KT"
    "^Rw02zAxSa2DxWu#F!TF5+8#Guly2YiIm@Qqd7pOG*(%3W%lFXD&HIp3-G&M|Hk|Rl4I4o%`=GlY!YDJbx8K8eumDJs$oh6s#7)h"
    "lj@5U3{qN>RwBZe)OCy&YC_!=yI=7~6?T_38k}F5+P=yL>Z{sF>ppk6z6$WvtLcz8kR?({9EMyXIHgOBdR%I%65Q;C4eK9I8~-%<"
    "-j51bwE*^_%P4S&BF4qmcf~O@$CEhdk|)ayfljDSx)_ES2(e6Wucd&1<`&{}6cN~{&>YeH(bQNt=!y{OjiA{OCG&&~54YMniz(BZ"
    "b`_3`I3hz1Xt+v;p8Rtx?e~M6{dnw;$C{W_Y`yfWZf~Qgd}3*BDraVo5~i+-RdM+YFr+VfSP=iJ-~-iCm3m0L*9&EL`dED}2{4ND"
    "cKSomG#8^dL_EW~C)(Tw5P`sKmv$^&LlseNWRJk~T7ewbf+K;vStIMO4?5&WL+UD`kIK<uAh6NiQ{WAi@NoO!mgDrpK=|?4UW8}k"
    "*hrf|hC>5=LL=(gRGKnHSEaa-Q*w1j-Y&?-5hz4fdL0>DGRWD!5mhdEA&WOa666fN>W7q!D27&|cCgWOnAc3rdyWW?G?>-3J}SjA"
    "G#K`D$4<5(=AsUaJa+jTajJwh7E!;gub-M(wejM*B$bF~dv6i`nLwR(9#_HnZpM!L5oh{VQLC+O)4Y(Sa*!ltpwR#F*ax&L{xNmG"
    "Gkm3TuOw;8r(5tT(P$eOQO|#To}n)X<eOK9>aQ2BE7ON?DaGw^NGh32x4#<H=mL<9*wf_5LtcZ<fT07jen#K&qhnws@nww07%#{("
    "EXEwargvNpay2*!Ol>}u;m3U;tjGkgQ5!2@9oGa!7Gw8TFh-!Ab6W4IBlU}oHLE2gjtn?|T4a|AcIiASmaBYQV`y8+LA8Ts5OL8e"
    "D}(zC<%rrA*KH_8na$K@ptrs84g-|o=|;wCyvlZFKnN%`C$pkN!QVMdpq?PWJI~X8@5&j`a=I?bMbv5~Di8tLS<^gVUVezC^`;ek"
    "+wliATPaseMHjtNNUwA2GApZDc`wtaIB2zODz>j>RVTWYUA8i<^a>hcT)tqhwL6_$@1RAqzx7rzX4)Xphms!%`Wtqt*kLlqm2UdB"
    "8=3Cu*%r%01ys5Kq|Hn;W=dMxga)gx9fI@OT9I=qcJrcvAV;-p?*{9=PvWr>=eYSRdt$`JR|2ZV=qi?B9jtr3Aj+h))Ru1iEjGyE"
    "&sbFX{H{U%kRYH-jiW+c1JQtRR6tSDfWYu?I8h4qQK|7;K@N1-yW;=pPfxE_w$#D;NXufV86{WFHc7`*wv51jXf1G{tAp%rzep_6"
    "SCaim%0yNd4^yt~Ow!Qk67AJ|Ax6q7t58D7v)+hX$fEHipv3jBM2x<)Y%ibMK5*}u)}5?rA|hA@G?|Psl(6(>myjz-J-C_4t-$pC"
    "%Fi^>S9Fq@rw6u|IhgHDc7A49Y2Yr*l0zfil9&y*n_w%Z^NB(oVI|iZ>H-D3$p>J2bvv%4dF<RWJ!d$Zx_b}u%kfKC1DSI>4hWzH"
    "X~~O80fkge_a*!{21LmJV;F6!Rg@G!6i~*qi`QkgLIGK#K8NHTog*i;WL$9yp`L4h9PygvydD%oG*n|jg+}J%uQP|G5sVsG;gMB!"
    "l-;vKUcHBgJ#+T9@6WTkg_<LDDA$EFDw7oT8!B2%{b`*5S#50&=&SQ+c^{bJiIz@CXluO2G^O@Fy^Ui2Yomd#J6%oJ6K32&DGaX("
    "D`l-FrxN#}1=5)sCa**c!Kp+l><DU9dXbvoBw8OG-c}qqtsa)1TgGl9Vy8(W-r4Hti6TnQBmGKp&<$YuPBN{CSI4|`$yds~rf!L4"
    "uW7pLRLsT#Eo8-9l!h5`Ri-6~@v9brbgcQ%QK$*Pamk|b(L@zOSK_y{3RLYwK^lTR78X*tb%ge?H{||qR{`;#qE*0<CXZ_w6W_yN"
    "wVaBvIDwGzCqo5ghdF2PbO-VWwyi5NU)M{M!G`manR>v5z`RyLOE;Szj(yt1N_XTzzX_LAkICx}ESCWs$PA@nc;aZmkO~atk5jJB"
    "UPYtpB;H=p1|{KY>UkQ7OABSyW|j_6Uf}eGQq=UxtR+g}-4yP9JnCB;zmOIAp~VTw(hIP6eY#s$ESp#AcO_MAVRIMv$AW|w_3H87"
    "=cQ^HTz1vmT^rhni$^@<%Z4|LN27%Tl2Pin+_`N9y#NiPU(v41D@kcROH+#F(-JnqnT0p4wI~)_zR=;O|GC;d!j-~}+)HzzT<!=a"
    ">*(4d<^gYxI<Evt_FfOfHkjSG%QUAr@8I;kNli>v&XAd2oS_x#4WR{Go6+SE0k1m{IJtUtof_P9q%Vm-u+{kR6H7)HKU6ec2^SI{"
    "2h^d{b;p}wz84{K!i)(87Kb7q*X>?*vm++?0koL!lYYsdEP~~#K$qBtut`e;m$M@#N8T++6L}?I%cH(Ied2N{i}CoPUQHfPte`TC"
    "J`OHk0hN&gnOW}lbO<YIM0(0hcV0&o^FiSwy&!ZCpC@6gu$UJ#gbNfGDtOF&RkpL6D_79m8@jGoR9xao@<4nAb^+@GI0MWj;`KPP"
    "GbphRh{*$JpJvhrUBN)H<JPN&VnWhF0Sk-`fUZZi!`9sLSfFmeU@1cuz;dF0)YjVaSkPr?$#--)v1Fw&%ZX)}UoIh%>(dHI(R0UK"
    "P6j5!irb4>KXEC7M8B}OwgGvH^+!XoL1eSGQi=>Ez{*UsJW{VV!7hX&$?gY7`@mAM51a%YDzzG12GUmLs2<e27;^sftRHc})u}43"
    "97@<sSH3^)(x;+mH*MC8`;@@GqZi6>(F}<?C*%`#;5+>a7$ZTO^Ow{1>=KM;5Hp5YwE((F)PVEE_?nBO`@nU22`~bSQo?PxG)ZzP"
    "fJr3gl|*=TW>n@Bk3~?#X{m80D0w1#Gw|*T@cLEgr6^iSd|Xv_P>$3djf5sfH>W<8jyo3TUEMvLWF4FRLe0XiTJ99-prt0f3S$yM"
    "(#4P<hp~`I0KDLYuf`l4pHLz=lz2(AW=wyVO(aX(3sPSzwlZ=`a9EO?))%^S;7g|M-yI#wp1^2J&{3PgrNU}Vat56=z>7@n`7&d!"
    "jT*J;DMle6;6JLO-ld~epgmc6n3<oNR8`d42!@z>b<~-THew9rsLkF9x4-D;YtSAoe&y`Ig~QLG>?v|~chHQgBMTHjQA(AnGEzN<"
    "X)QAiv5=%a%EM`+0#Aw7SL82>6I}7Bra_i}3hZEPT8#o8v=JE0Urt$lLK=+%A+)iTDpf$1`luuXN?9A3o?k*zE@UBq`EBL$#^`fo"
    "IToDG*9kOTMvl=`(MS`=-9qC<IsJrGkqzRYteI>p(0Ci~)rgcC=9|h$o00TW$Wd0<%`E$Ap9=9wWC0^MM<xGUE4$fAwUoWkV@@s@"
    "V`|f_%rw=h+$dE^5GW?a<AqDxD7>XnnuP6fftp0g1%e=@rydGulXRXK9U#uCF{=m5R86Lk2X-OGu5_MzO9?k6D<rrxrn&eEdYvkN"
    "p&SX_CL@};A1S{;q#*~F*6oc94;Z@JMXzsBQ}>3Asm0G9r|$2iK<W1tQRit8XcrFjcmbL_IgvwueT=>uurNp%MhL<6dacDIgMraN"
    "IiRg8OdN_M=5ZWQgrhrJ%1pt7Dm0=GP>1uu98P2y5-qP;Hv8AHef}oPKBR#_;&Wz;4Vez{*`S;HtdMqtfk|5FLUuw@O#_+iS)G%%"
    "McI^fE+2GqW}rzPS8qKE%?IoSl$cH+-8{&ZalCW80)rR6t#KG2Z%z78@fjIpD51H9LZoPnuQ!Y#vpOo3S&Ezu>MV`dRnpt0qo)=y"
    "A2Y^cNRacB_Jgqop0=U1@3vgt@75Oqwtwb~KtA5xi?MvUOcF}VOq`lQDcQ*UBpUXbK(pa?sWje_pA`o2>P+Vz)?a46!W%V<rKpkL"
    "oOlg($mZC^2ktHtFeK{Pn0<;k*PZH7X*A<&v!QjSCMPWoO~%gGL8_95yZZpn{6c^zHS!sO*{MoF>81u-FfIR-<<;{<{Ba&IiAj<V"
    "Gh5Y-TN^9LV)GJrlnp<i3T9kopOH_NIrTBA&CK0DMXrf5Oj&TW;+W-8ICq6_ofx?1;New~t(t*!*nWRl3ZjeT!o}KI9h#P!n5(RK"
    "p}5T(Lp04DWqdwlsSeeKD~Txd8ewCdQdLS)k!wR`1dmp*#r9%02@$GQI%LF82;G~ihOknwtd3tC^_6GG5}`urFG!jKowaxpC6uw)"
    "F|DXQP{2o0H?a)IsZ(#>{!th9;BZXGZhH_J_FxmDw2zj+&Y0i`Wr%)$(@8BtQ`fKj>ZdH$ini)h91t{ndH=me)oax9G4XYN(;7L%"
    "#mIUF9t_EI?FhG5Y6T8qhj*hY6;;K~`JMQrbLa17x$9C?9>AVyh=&6#BLnezoI6RjE{IXJQFj*5u3~mjrM(b-=fwnRi8Ps<r-$TO"
    "b@T`dys?ZAnMGRQL_AHbGnPdds0;%+k&AJACS8~Kntdc_R>Xi67&_Up#8py4EbVegl4A&E*sFkh^`FJe66TA=zw!k(MV^JA4<$&w"
    "4bcCTqU2f$)3k6#ZTvL4f~Ocr9qZ0<^fR#4dKZGS0p$uv8SVMztdbVQ;qc@Ied_rrQo}lM57(`GxcNPBu3sSUIA$yem7Yh_8LqrT"
    "Dc!W7(O#fgNzsj{;#eXT$BHGrSVl64Xb;zwD@cU}VeMkxaw=LT`{E=UZ%lHjTEFCadrw`IBz&n?@Kf>PRn1w}GNUK6nK{?xsKl{b"
    "O%32-Mh1Go0ALmRh9_22uT7JBt`4`EVV&tb9QJmRz+3PSY)362*+k?danv_Xpc(8)o`y_`60ty@-)f?B4=w<8z$4~O#3NUwd6ekB"
    "R1?uDPkRswo{k_ni5VO0#E8b7ktBV>4LK0`TBeu<tq<uB_=95_T#p;oE+p~<(ddUz*N9GVn}muzC<Q4HXz~?;6OI}!)C=J`;jGIy"
    "HeX0ec6wLG2-Y_ZH%`~g3EU(yY#?q=Cz)rvQ}S;I+C`d8WA$ybS(n6J^e75zwyE^AtxTDqZEO0cYBxq<MAss8PwayG03`|wfIb3t"
    "!S9P)@UmI_E3J&#X!NL(=uho)BShj0+m8?dNW0gSN$HaqdJ&e)Gp3InCWv298vx;1Uk<>?QMsKyfxJJ)09}$!nc@KZwSOf#3QI&`"
    "!tmVPkKp0BNSL*_TW46Oq=<SF^-PGBlXY^~$CM0+@F>1yfS9UjGb>|NBVj#sH&L{W;}_-n3#9H9FKdu<t2ja27;6Xlk%vx<!Ds>?"
    "RS#H|f)KmMP~r}iYmtRQbzrmf0pKh7tnK@wp+>>lt&qs`L<todTr7=54zZ|%8rF}6PR1lGp>{+uDo;PbDPh>(dWu*i1qun(ztHYg"
    "P14^cfwZFaLtD+5JZbK**qK|8JQ1SgbqQKT2E%4gWuT|%#XbN3uoLYAr2SB=9kAF&$je%;DzN@`z)M)(`5OT***z~PC^{tsMiF-m"
    "m$L>3gxQ&bOZs^1BKwquNof9PRkcJ}M0LJS+P<8cqTwf<St<rwM`GTF<*_{KB^6$el*yohtRpv$E*@DtrBs%^7yb2$bCPwys1{;S"
    "P2x(`1l+S#nsW~1lx>R8RVLq6Q=x%!;3w7Qn>L1eW9fp}cVwx_br`9>Bn=m9I7AdDG?9}8P!M+(uXmZM8nnw)uX0H#4?$f}k?e3b"
    "3k@VwfKGXu7u}bWV}!MMWfTg`Wh1maBP>Q7Wr;9nCR*9;DygPviRETBz+{?UAL$qh@qMOFl9~}r$VA@$!klc6W7SdBawlTitUf1t"
    "H}q7eOm~sSUL*{6V7Ih-MNE`}3nCKQ$e<0=StEq}IHV_a^=o|#Don_GX%neVjGV>yhsb=dnN--b*DG42*u~&fDnq$c_<us(DFKNA"
    "AsY*d8KDd7)CVjg7+#)MUdQSxQERQBXl^kN0`RW8>WoHVAZM8k#Fi*#qo-XVvoOx!ktFVF%L(&}b6s082r4n~bt5(qZgGT@Gub+l"
    "S=L?O2l&4Xe~RpLR*%ye@U`O;Q7<RUD+$(4ty)U6D%z_yN!i*<r8fz{XMsEbgn~eO%t%To|E+VXNTpDANrOas2^0+baaQO<XywkT"
    "6TsB~sP=@SuYl_He4R6vQWXURDQXN*r;I$X5SiFkwG&K?M?{B%a~{tL3oKd}53?yL(@Rl+MLloVWuVNG04QL@v!fXgWqkfHl+4-+"
    "rpGQOQ*mi0C@315#);V#M~tX3d>4@DazR*<WhH8&32Cch&Zta+S#&6)vU<XzWx`QEAx5FF@yb19IlY{v7bbKzh?JsIU+iQr1U1E*"
    "mRz_8i2FQ`X_TM!4U@47HH*F}Cx{ib{&vNQ3PA@w*CmpT6A-6L9i!-T6=Vwxpk4=6PTumqPUI<0ckt65UsbtX1m#`@M|7{EizG=}"
    "<f&zU*9fW8%v6or#xlsnik^b9ZkmLciWEjchvbq1S1tg<66>cpr5SP}82oqgF-6vG>R<UJ4+gOnjYK`9jVCEGahbmsr3V5yKod_0"
    "?80{b1SgcOQyvf2k-7rSeJe~J)tpYJ$<1}@bXo4LiCb8yiOhRs+mDuUj*Npbb0-0XIth`^oJ6CIvMr2Rdn_dxGdUTeB^ONDp)g;&"
    "(m^TH4AbJr+?}o|T~P;GI4B86kxfr>*~i&5(R7MegqB^iD{V|%(S!4-Lj_2^PMDC|+Os#wf>uaVT5?;Xi`C5jw&R@L1S}in{MBG|"
    "rRtn~3>Q+a%NQ6@A|To0z!u60q6wxe)CjDczEoq4r20pOQIRdHPUXX`OIdkTPO2^L7r@=VU@5C4zj1lr;G2)Zbr9J$zeBAl`Bot3"
    "#=Kq20MfjY?aFc_dDoZY>3KcU+`&yxcoKRV3yDp)^eq69<tc(m`jeL7Y4M?GoYrDDN@3w!RpTP;Q<hH{rkTZ(R(h5rA;d6d3=2tu"
    "n#W>6Ywc(Orz0uh+a=qrP?m}-SrxKM+4n}uZi3qC3m9>cy6L!##MFdVYbT(rpf4h*a@C%``Z6idln#Eb2;6At*6R6Y%a=1iGAU!d"
    "L>wAntrmIUil*Z9s*cwnr;DLo0X)P<+Q^bDf~w3(%2V|<5$R&}InlKt?T+HVOusrxtQeu^OivGRGoLV(1~*_tvZVZ9OKH~9y#Td{"
    "zMS_O`U@L@x$Dv6W7p`eyBHc;m6JN!bB<h_Thes>KuLFD>9S;+qeQd@hk0r|$MYFnf-seEnrRp$srJ)3%gbo*emOBezE2A;p5~as"
    "f#Dp}q%J2tyO#{g_A`s7fv?~!uUXj|W&K7$`|IvK&)EQ>^DCg~5fkZtq-TyTboYXCSzUGRSd3>-9reyachL`wlHjh)l~?R;?UMKU"
    "d32qESB27cDNvVrEX=6+I*hCEwi{9I4i|L>MtADm+PPNq7y!$X_rrfuMLe#7QBQxSKuMaR;@8K6$DUU2DHCrKlT<{)ol-oHSq@ZD"
    "hC3x{^lPCkIOZm>&@3a^v7gsM>T8Emu<Pt2PT6W00!t@!IfUP2oE2i>vV$z&6kpI?QCLOufyi{9`wK2?p7S6r)`QC#keEC$rGQl@"
    "WEYY_;!*{hI_O&W2DYTb_IC9K2_h`yMeTy%{;J7lQueO~OL5dD$wW!2qN=Bn&JtNVe5MT3Z$PCp%Ar%B5%OQ`u)eg_YMabV#WQw7"
    "aO?u9*C$hBrv|ecLKLh<Ub1%EaX7<L5Jrw?{?Nmt)u1L({g5N{r~b7F<g{CkoW+nnhm%V>E!ELfFkq@;K!Rw4d)~bY>g8ZtFqQ;@"
    "M=tW!t<~g0d>2$v+;w4-?ZiN;7W^{K+g*+q^Y=0in9J{K=c5i770g=vD5<$-$S=u4Z~`3=Hvrl~IkjZ(up;V8Y0YNX91HSC8K28T"
    "&T?H?OJoYZQmm9m^}MsiU!FQC;4CDnpiPx9AV<{uR4dtwcEn=|LZZhOR|R6kaao(KKP^R(o@|e65pp;vyUi#(RLRH$+bng*G%<DC"
    "KKCX@C=QuqmK&u+rQ&I0P$4O?4fm#qcLc{}9R^vHx=Hgv^J|?GQ~m7pmAM-4dO23vdP^m(1nOG=5@xf0HM!fK>_v&PL0w?GwBd+U"
    ";>2-k7qc^&jHI8>o!GsHzN2TFdpQT#NfNYIR<f9+KXft>$R;W;4DAIB4m1Fnq{~Tk26LEL?$kp~hDFPh^lmss7q0@8uA`j3IxL>("
    "8KyF%`&5n5sLb96CoP?lqp3=8OGs4%AcV~CnKA3yebzwplu@E-j24z}M@FxW1BNdS1^jm3j*3{Q-3gR%2htNk!TMChhmz+cW4G;0"
    "#hu4I9r8^ldrqWV`EnBqbFhlV)QWy8avRS*kC<kgKH-=I5dc<RIzo;lVt&;0Avv^^_Yet0or(*RTU|8{$^RlLV2oXoK|(j#)}fZl"
    "2r(JXs?p;1SyjnGh<0j!>`Is`1R<%422JUrIgO)H6i0?LRqxBQRB4*+fC;P}zlXi8+r<&|K9$B!T|5%-q%a@z{Ss<eA4xBA@pOTM"
    "^b!%F^phh&bHr~hw8rgKu5p(vWs-l_J)iyikd+iclGzf_&Q93Dm*B2UYH80{{U}m0%2h<x*q<u?-mYe|?x{5ML@;p-TYt!{kp+=L"
    "DS*LePJN*Y%4eB@2N-x#_1OApR72vS927*(Xm{t?-B*#FMfF@-jHlW$E|M500cHY;l63%xi%MoXgZre~S_e*)4s=ZA%XD#Tt~W_S"
    "7fZ(b9ErsXf}&A5ldyKl1HbN0J1DH(`*0iP<OX4-8=BO;87>6aqBkt-DNQei8Qv?c47!uy-I@4k$ov@?9P5x#Q9fy<jt<J&WM+ai"
    "TOqpA7z;-ZK?4hgTmd7JLjW2@I9MevJ96O^7wMY_jek~9o}yK9uyi&hFX~celH!pClm!N3KFs3c(o+8;VsL!IsH%F3L@)@hRruya"
    "12EAY$lNmCF=<geIEt286D8z0)E}^#31a9}s!^mr5Rq+ZI5!ek1NdbbO_Lk7s`5!?_NibcFwtdvPim;mCufx)WsNn1LTV>d?rNMT"
    "&g5+A<RMuzQ@_wqiz_E$(@``{5)cFnu668r#StJJtoZCNsX9iPK6*jJ`6Vu?miV+RZp1J#33z8P_(HhFqRO87YGcqntvSilj8mP_"
    "^uSov@dqY<At6UdWSc-QVQ<q;F1yjdrIC0`nU8d`xqSQZTGq=!>9$(|K01wWTV5AjyicfDnb6kS#Pg0V5Jt4%xwi`=P_!9L$fGno"
    "p5&z`CX&43t^&Gh-ANjK*IgtfK`yNwiCdUDsAf&w3eZw2ZCh9ke&u4F#z$S06eG&hBv>-3S49~t=L}t>NK)vsaz?C~yKIz9&?@RB"
    "iP(a&?%GK0(R92qGlSrVQY-H|igSpDHS1oUOQ`xy>taxsR3R`}UvN%h&O$M@U5U|(zyjU1B7-5FlcDA5w2&--^AL-xyCn3ot1_)D"
    "8zxb;1d>b+JKRK>i5|QZE2eW>_r({2FRmth5D<=`5DG-E2VvX6Z?m;{`eDO%%)pUeP+ZyCJo2DqScauUV`c^)CkCuuJ!@ItO6;VN"
    "GEL<ie6euXTvix%V|}btL6WpLNCL1|YLoY%#8S))^d_X#h#fYf0}5}z(l8KfZA8<<<2>a$r<zUykqi;V9}CSNR>*Xz6POUTbkCAo"
    "%9ssGB%k$fh3`iAv#}a)CZNOxTNAS}#WYC7T2>CLg*OE7Xl10QIb^5|G@!6>US*;~jm;+7C=iW4;LEZu8)Z~vxkg0o#N%(t@v<a5"
    "9B29!`D7e<IP4WEfF$R|2=8l-PgZ*}DR0z2TOh=k<*zyOm4J;|&0n7UENV04&T2vj<31@HQ)SXXP|;L@6ed?x8jtL*WEwN<i<d57"
    "tgTMMAV=FZ9zR3JI-AF>QsEpgY8F;nn6any9!wCe*PlMQO=Lthlx_0$6{!r?`De2aR@<KcuROt^ZRfA9_(`{E9Fll%-UOO<xN*Az"
    "xjm_Aq$gv?EPDPoo6USJi|y!a+5CE?P{t$GDax1?8(Qk4MfY4M(zx{~xvD=ONGn~zNV!E*Esr-Rd<LTdEWVidF%*Z&-%JObWS(;c"
    "$x@Fs5^@?{O|9U>>hz?mZJQTM?iE<VY&Uv70LVc-or2m|r7;JPZ2k&(qSu`;^F{oAFidjsdOD1FE5G=_2q<@<5%XvtcR@qrZwehn"
    "F(P7^6y6EC2_$Np7a-AwS;NRn-+X-kA+`FYi%@cqLe0c=Ziul8m>Oj&9}cOCB2yz!N+;{0`&7t_23+#4Qi99#vY32ef17<^5f`Ai"
    "v{>iv^7#$zekXrY%x$9LnXJg`u|59x{{vw5PSp"
)

with np.load(
    io.BytesIO(base64.b85decode(_CHECKPOINT_BLOB.encode("ascii"))),
    allow_pickle=False,
) as _payload:
    _CHECKPOINT = TapeSelectorCheckpoint(
        config=TapeSelectorConfig(**json.loads(str(_payload["config_json"]))),
        feature_mean=_payload["feature_mean"].copy(),
        feature_scale=_payload["feature_scale"].copy(),
        train_features=_payload["train_features"].copy(),
        q_values=_payload["q_values"].copy(),
        global_action=int(_payload["global_action"]),
        action_ids=tuple(int(value) for value in _payload["action_ids"]),
        metadata=json.loads(str(_payload["metadata_json"])),
    )
_TRACES = {
    int(action_id): trace
    for action_id, trace in json.loads(
        zlib.decompress(base64.b85decode(_TRACE_BLOB.encode("ascii")))
    ).items()
}


class _StandalonePolicy:
    def __init__(self) -> None:
        self.checkpoint = _CHECKPOINT
        self.traces = _TRACES
        self.feature_steps = tuple(
            int(step) for step in self.checkpoint.metadata["feature_steps"]
        )
        self._common_action = self.checkpoint.action_ids[0]
        self._selected: dict[int, int] = {}
        self._snapshots: dict[int, dict[int, np.ndarray]] = {}
        self._last_step: dict[int, int] = {}

    def reset(self) -> None:
        self._selected.clear()
        self._snapshots.clear()
        self._last_step.clear()

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step < previous_step:
            self._selected.pop(player, None)
            self._snapshots.pop(player, None)
        self._last_step[player] = step
        if step in self.feature_steps:
            self._snapshots.setdefault(player, {})[step] = encode_selector_features(
                observation
            )
        if step < self.checkpoint.config.branch_step:
            action_id = self._common_action
        elif player not in self._selected:
            snapshots = self._snapshots.setdefault(player, {})
            current = encode_selector_features(observation)
            temporal = np.concatenate(
                [snapshots.get(feature_step, current) for feature_step in self.feature_steps]
            )
            action_id, _, _ = select_action(self.checkpoint, temporal)
            self._selected[player] = action_id
        else:
            action_id = self._selected[player]
        trace = self.traces[action_id]
        return copy.deepcopy(trace[min(step, len(trace) - 1)])


_POLICY = _StandalonePolicy()


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    return _POLICY(observation)
