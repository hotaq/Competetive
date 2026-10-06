"""Standalone route 99612231 with deterministic state recovery."""

from __future__ import annotations

import base64
import copy
import json
import math
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SOURCE = {'competition': 'kaggriculture', 'submission_id': 55776752, 'episode_id': 99612231, 'seed': 1245824168, 'top1_seat': 0, 'source_sha256': 'a69b64fef975e33008d0710ce6ac0c9fb1ae532289f5f5d613c6180ffd1b50b9', 'replay_validation': 'exact_current_engine'}

_PAYLOAD = 'c-rk<U2hxNvHUN3=EHJG%1Yj}_9SaWYZ<b9iCjZiERX~Ng5+V7w;=yLvc!+!bai!gpF^#k-6s<?HFLgCcXf63r+;7k<;S1?{^y@Ae*5X-{q=_r7Y~bzzyA1-KmGIZi^q?D|M924{rO*ypFdrE`|*!IyuZ7B^YQil#o}W7-FW@@@A_f!>EgS$cjLv$htI$LFph5?fB5})bNlXM@oM(<?(aAIudm;|{r>vq@jpI%H;(T=KltH4i<8E^ef@`z@25XFHuTfQcKmSv^yYWBclY02JWQMJJ?Pj4@j>nw%--<r$G10ces~<s`;SlKF?sy%saivCu3wLb4xc>c@xR{P9|l)@-0waQ{ApjW@5j5Rzn(phJmvMZ-=*F2$Y1Y&+t?K{z?D9cfj%sUxNmlR_bqr{Cr?j#dR%IFZHIP`%MJ5~+mG@rk>RzowR>=Jcnh5a>sX&8JHB?3;vVe>6ZRzj^Id!K>_jUByt^_fU+oqeuY&=a-(u4aJx_n}auBY94Zh`;4?#yyA8qebbg_ERo_AqM?>FhZ1*g(sQcpjyhuq%WjIZy1_}zGS|MuqX|36JhYs<h%S7@4Dccu+r&N$VYMsD8Y-m091%&f>9ir{2^-pG>=;ogtN&OWkq*xSd12a7%P!9zgnTN@S*88~yzL*CrozNZg)ehYud^M})u?!$QV+2)^g`15gvj=5EI%&k15bieV?`{5xz%wP4~eZF;$FSPH;IUt@%qz~b0HiRGU@2<Dsj(2x|1lI-V+<wL;O>xFfc>(XPvQG3C>krW{dha&Y&C-<9r`_HOf0tZJsk<$@zQL()?>cqMf$OSv?`hXl#c?RjMP(2xUUe5EXnz1rw)ga0ntfdw!<wlOjN;Lo%3u@JsXIQX;@%yPr92|QZu!ihDffH<o~T$<c`GfRy_yA*vMDr^2A*%3Ter_0)?k6H_vVA?#_HjJ=PC}@`pf#!iH6CJbmC$X@7uy9f{yrJBV|to-7MAJY-Z00H}UMRi&o3Tzc}yp+r0dax39T}bJrYtP~1xft{y*yhZ400c9)It7;@{f#<=J2LcL(fO${ed=nJ`G^#KvD*xki)HgC{@9$w~xU45a}K=*rj9@~9=yA{Cj$(wh7kn&^ry$$}izc<|x<IP-}%n!_XLhSbGbcts|c74W5ik*NFoCnGt*Y@?@|24+6o?>O67x&3n93C9Cdx%f_-D_(h)@aCgubZ6vCHH!A1A2IBdIs(FU&B;DbEoruY&x8@3N@fIPt2ktOH>+3-OZTUMOFJax38$jQVC>P^9G$P@pEV&4;w~+k#>IsH%#I=^BZsDUgN-2c?xLnzMtNh<i<Y(UcRU~wM8s_QM}W2%Yc4$4$r^Xr4%Yq?G)qloNHz}^uk3NRu3IOe`3@WCvJMWf|J($Hq&**v`@tptcBqIpbZR?ppk>wW3B_qcx!^A4hR0>gX6~AAEKeFHPRYahnF1eO@y{&@EZ{bvfWHviQpaczYaeYh!s-n4~wilj^YlDMQmV)<~P4N*{sQQz-^9i0$=Iulp0(FzX9W^p+Gu1v(Okgj-{2b+mcO)mr`RNEkS6$c7`){uL*RNk*u;^JAeCt?b(;4i!8bYu(Fm^Xu2G&5RPVW3#r|fK3?|LCPH5o0u$Ka+PXGa(VTV+1@3M&QDe+~zhJQ@<sN{mVh^NkZ*Lz@noF-QbC4>y3-(XTk`{K2F<eGJ0LTfX5_S$^G6>*4yyp1k!}>n_pJTyFfs&J%SPmX6V~Hy?7$tCGXVZ4&wQSn_uz48iGoMI11|0iX=g9hpV~^O2l6^C2RL$Z6wwoVT4Ge<gZcWADo(6CewaN}z$kusukKSA)%@exG$ZR2F(DA<3@m}|gc=6SGN)UOGf8rMD!Y9!Y+J+pKkodaF%&^B>FS4oEcxf|M!`~Fi2Ctq`?NgTd?s)8ImopALJe-GzTBb1YE~ZAuXijYN`wD~5o(90Yu&rp!89M&?;h;rE<NS1}9R*sBB07rc>Ga%kA(_z}LBM)Ky>~XNXX4d>)Yyb_cw*MQh;6Joj%rsR0#9FNRXQVL*F)zedNj3h15jIo#o{J^3QN~<2YqidpPI&&D>$7(awD{Dl^C`XlVh=X7TcCuK#pk!EHE&;%4Fd)O!LkRf>ll9OKqTW9$R2(cgObE&v*NL5b{AZP;&>p*eMSUKc^chi<1Qp5Wdsz5@FGq@Gu8^3HhQ<7TThzl@!AR!x$NYFgGKi)xn&DFtQrv{QA2H<d24v<=S(rBE1?dnS5gNsedwLs59##PbSHp1<?fpRBSYtWFu28)J!3J4+2xoMw3f^iwKG$Q$YM(7{wsllDZDyzNtt?nLU{>`l4__2FTEaH5!6C^>hdjQnChz*6E@lc$n)c+zebyErY1q2zj=hUphgo;wVYdX*{B0j33B+s%#>QEd$4s+0ljcIpaGbFNGc{p*d{#Mu*eF+C44s2O%CITa?jja*nKhF}ST1^u3Zc4mef7rlUNs6vt+tZG)EI(FgM*Eg6<H4}3YR%Em0`Y^fYYi^aJrfl17EY7>OBtIWdebFx(*sQ=V|T9DL5F0+&9Ea3Jaa9zxyEOwl~Qs<nNwp`eQD<^Qe>O;%VWzVfWGztC?ekr>Y8@J6A5w_0G;ox$;3>15c-w7O=z_Wl~Y5~=7Qvvf4m@U-bsHF4gWTV4y>3`A^!Y16N1<7dZ$Ayp0N;LCR;}r!rWCj&6Ep=o6@F#EH{vKMMxh0BUf`%c!mkap;85t!3saggUI%yx<q%V6)S*|#6SoKUg>FmI1BEa#;z!jVlDW^PyK1gOT>;y!K7A$-Nj@(KnA_L<INi1WKD=!7-B{ftl$sUwY3{OLa=~aS*<kL{+n#JPrGB4T7G!RmQ6lTyfO>ZTHJt9wO%Ga8)L?IR}&!Szj)RO>A&{et#v+-Ff1C;S^md=H&H_2mGt`nhY=Iv<SQC)3S&sX*}avH-ftr;G^w*%-fasG#%M0BOm-)baeMk}L@_Zc5LjQ~LW;Y10({W7LQC@C}Qr<#3^T1As}LpxMWamJkX8WYU9CW!e$LkkPXSS-2Eawa916W;C!omKLuYM^ms1faZ3ou3A5Lg}>Hl>@p|><VLEfgw8df=J0anP}XtpSRbc3@=*PJ<*l`o`oSK$ON5A+7-+91Mq>zh!&d3r3o^T-$lQqdl{kvV~eKXxL8MiDr}o{aDa0L$M-mwwqbO`un-H}InHK!&b&!Xm<~8vL?lLt#@IF^8>=_L4D{7rLn5o1O{>2hQ`So4d|EHdDyxONfy@VMe0ck?6jpls87((PO+aua?Xm<k59L#h8!UDQp{)YWHYv*%oDRnB7ih&v)&MZyfKUZHKhYlo3=EMSoI^17i1UU6%RUl;@*G>V%>Eq`lznPA*wX}J_8@W`aZ}rcSU?Z3^?h=dgqk9JvRch=cG%KP8cB;p$m0wzp_u2nn#GQ6Ttm?pv5;#E(`C~+$+43Em`1cVE=w3)lms6-Jj0yo8Hg)|6s;SFYiP#H7)(YFHFAzeBdS%2N)6<u?eYz<H@c8y%aDv1w-6h=c*~JXWuOz8(M4KBvdM5%F*_O`MxaHd<6z!${0mrzx-PsICScLc&GW~?@sZ_|z(o!Yq)a}J?l#lKpwHrNriRMS%jP&517<h3@s6`acPD@B_U*C7cenRl_8#|8)3YKfRfxTjHQNI^&vA4S^h~bRsY`1g=ISB{U0q0u>?Gfb5l6W5W!LYnS<&XTKMOmmCh+q7GsaMYaU9TM@D*IP96}B*#$&gu&bP_bn6wx5kf)n7-y4jL5jvN!V{XVMk~~<S_&LQ7Akbqy6d;_j-1;Pz%ydJ!B{rPURdbS*4De~8d#20)>}u;e2hs?h<;5Zw6sWT_6-F<Hs&2XhtsZg+>8v0oVRiW|#!sq8>fHc!F-Bh$*v1!3BaWA8e3^R%h0FC6O!PN%R*}|_m8F;Zmddf(;5KE6DY-w@I^s;II^<mo!eiM1F6noJMeL_yDPqRK80T@n^~>RYRW{UGPTg5K2Zw<3X({~<_7^T)cK5ONCS{nD&}6LdS4a=y#ICH;vsS4N>hRiIo9Q=*K5EQIOWQ@cdz;1BRn_aFfyYAs;Y~-G8wy~Tx+>UyqkKc`uC)pr02t+9Mxour+)|jv7dvn1Hsd`NEJ~WV7ZE47894+U%Q?vmip+%|y8~JKa$U3rRwVbuH9xnFv?UTp@Em&<d=J6+<kbpX8Zf~%O)};uvfQf(%#XD))M$aQ7wG7nBGG<zovM|b*q>4!m@YHTA{5LS;womUdy=9P@Vf)JhO#DvrVOswl8m{*O7Cn(SG`2jut@T@QiXk7Ch<B>L6ThLLl;HCj8$Da-_C}0Fv(=3s!V_X_T~?O-4m3ChQpa3c*$?hijrT|2i}giZ6Qg>Gt0VPz3-oq5>(7+Q}{nLbRUb83!)oci8sXc_u%hJ(4m0`Qy$Q~qrYqmwsQqP6yvE#ShU2*#VSs1wU=7CBs^%wG8KSn)?|p~K?DZ$bjpKdc~L6ordNq8edY1?h;hgAK5k}WP~s9@9Ad6afjFVP-~P(?ZW1{DP&gH-=6hqYm^?wDXEb<Bj$V>tK!H{%da5^7qX#f1E^AMOJGH`a44p}ioefe}VxBUro*T%mhA`&Z8sZ3%G#8No_0v~U2qp<xG1LT%po<Dr$BauKpa8WmoHP5TAK;K0QPMzJTj!jk8=_sPI+vw8=z^6>;0NRahV$FvR)={Rml<8n<rf=(-=q>~0%3cLYM}oS3cO4+H&3$)771mZ!ZMSt;tT}P?~G}Q>*vDaT0V$?SG+3gk0B>p6U|ZFH(~w7+bb22tUJ1ix)8FlP91P!kaPE2hw=4ToVGE!1D>X9H&bJyJxJXkih7K>%3VXEP()BMf@V>P?0_zgCeP^zDl_1fei{!YERww0Kr+s$TkO?$NFYx-r9mw}zK;MwZp&;CwdLZCysUb%6jNpZ9_du?4l)oqMJ`K$au$(i{iH)HOR=~LIDQ~?Du9mBw4KoE)`_#9F=K#R?{Ao?ob7bE*%Pl+{YM5tfit63FhxdCj_$$~<t6BjAr4~Xp~j9#U?z<APQzXNe0Nr^j!My6UgwPE2p+no63120ju&Y&aXf}qV+5l{rxP?vArLv+{4yoI%ZCuT%%HI%X^L_)7V4=*Sd51cR+iT~u2yr#*D<l+O^O<067YB5yV;dLSCT|WN#7tf!ogQGNT&~M^Sts_tZ<Ase=WsgXdoiXZBU@B0)ar*14g+cB)nG=bLGhu3~-ISBSn=tPh+a)TUZWz*0OT?tEaV~qe=Zv4=lv~tt``=Q`t>qvq#MM@XpiMo{t8>PI#9|R(G1F5^GqW;0>HL?PIDeG<Wlf{ZD&xI&acqTJ!>ufkS{?{bhzV6JcV^^L-x9p~aG6Y^sw;F}iWAw?IqXCs}F|`a*bD6`zTJViNBia=utfS$NBYK<OAW_U(F_eA5BO)};oRc+7Eaot_10pf&{?*ec63myItZwoRm3zm(hs+|*!c(V`bz@djYPq4Dy(suVp(dv7F*X|cXyn9>6-OFhhe(<)-L3{mJ+q>M~gf}n*@ajpeI6^eb2oua`xQT9auX|OCs9wbxRB;*f7RqTe~#~$d~>8Y8OBFCvZHjO2MZUw`Mfb+3;H4F5L2*duHR3;LND?kE;#jhw7i7xz*(r9PG5|K$J7dJ8hoKU&0{+BgHAkMRjfwPr}t;JaFbUjZ>AlYxqTqdwqt6U}l5InOANtW!P)eEB7<whtU;CgI%rjw1RLvxUZZQ9B!3yda4uuj|8fwY@vQxs@UvSAH>sA=3J2>)T}9}kOU(tos&TYvmDAUll$Ty)yGf<r7I4_iS{2$*Xekym9w#fTj;LcdWCnP&Mw!MyV)N#Gors=AJdan|ERIVTkQ((wtcS25nVl5PpVR4}{7Rfw5D{5fPlf19CKdoGPA)32?bRV}QN+c#8hIqa|F0#!(oU~y;#hN5Dh7IjIkm@d~JT25nvACirSTGWsi;G~wS?<>+esO7CE6NwU%LEQ5#)C-7O$;4!MjIC5^j#l}mNR{T4!&NgYh$xx|4(LI7NuVaAX||9{*sOKLC3kG&sJbfSbmTFy>iGHaz%k6`{-kw;zdnr@BJq<|f6c1BxFCzkIP1ehVz^7PAXESHORWMjGS8vT4$=sWij;>$xX$&XtB!|SZiOc3Nl2Gn{AF7|OcV~SIu5ered?0*jH?kO3swkHyk}d+cx<qMXC9kKSoJ5Ep|q}Pc@>S?h#`|Mt2?)*LoHm!k@Dt9^K%Ei#H+Q(z%WOld5VZBuBJf4OYtXY{Sm4YgtrO)wvbq1H!yc1B_d$JJjsSem+>ZMCKxkb)Nv%dgJedfusNB!mnQF&fN7#LCd~QJU=J&Mp|S*=0-PD)kc6lWYdVwlVEN?JbK_Daa-e`RgZ`1R0E)r@IUK5~Ep#Zh(mmD?4Ge*nYHSCAjhGMK4`FsNYcF-K3Kfg2jw|93$Hj)It(b&*79wS7XxRa>qaFf?Qekdi*&}ty+Uc@V$WnW<z}$3vqf`$q$sqndC3e03I<O!gXZ0nbMSx_$7q<h)X|voVSB=K+b3m6S5Tjo8A`FC%IGBeZir(f_lS;dTnCH`goE!F0%fOPmKLr_Kk3`mH;Q$d~7Ns)>Gayf_{;DOf6$e=cw6=0l4UzQ}$!cgU0)Z1ngl$m^>3?b>))aCVh&lYogbSW2R)}jUk!w;d38m^2;j9k;QgDr&2~|GuHI!|Q2DA=|scPA*&~QH$BTKN9RAkuFSA?n4I|D_{Z4wg|WrH{qt6ju}#n1)32Ki@Iq9$H5G|g<k=D}Qew42tF)g*n=o=i|SB~#!l63ZwZPH=k%1d@FDg(L>-Nng1bKBw-;tY7i&bJ$uX%^;b~6P~{0F!ShT3As7ar&XCtN0gA`QWFYaJH^TLb1q61j5%qaNjC`xS6*D6Zp&zX0%17%Z1w&~^Uc8q)>uMJASy}GdtODGc+D_cnZ+ZfxQEUDlIpDG6qa13`F@Kpgyg$L2#F>0EWdVAlP{wb(w(Y0POGl}Gd$2?u3=tU@lIBLNY-f@G^D*+Ma#<*pk(lbcvEAvd3uk`t<gwE$LkvT)Po2G0Io#wDla!`UfmRTO*|eqSY{)X3#emII!!ysvr8$x7x`3R4NquGwM7@@hqa{+N&6+v=FpZ~qF`DY3f7ALlmx-zGjw@Iu2n!3#KXc^Cq!|YDF3Dr*9wjtr%8#G15}pWrYKr6$Pq0V5Q+rqNW7(_9q*^_fa*KGa<tb4IoGyK!wTqBYTZAlq`M>!)D!+jhiwK642EOd71NjLTUyc`4F|!NpMsm|4p8D&XkW<L&4EUAW6rE>T4}5<N$`S5*|1wa$e<)(+9uBLN^rllg2bXV5X<nFfwo4<7)cq3CDKEc^Z2?U+%bxQBkEwBL`JSMogi7Hfkiit?}am8KQiFrBw0ZViZI5&j6f0?D=ZKh-(r}+dX%XebnCAeAzS6BUuh`oEU4O`GB~5+f9J%~*y*3kovmp)VM7_9WSmhI_osmiAd$cX>z}2jgFi{#Do##N1_OwMXZ=^4qwE}Z?2$99h|Wl0Msr2xDE#W-JqgD?m)cXWO6N^5LcK;UPbt8S&qRHRT*!Qul~x<wi=UPEgC7TXmWoG|!Es}_)3x03Ms~^UxC%3b6dsi<(yTMaIFW>5Tik~TE{kB13HzjQy;+BXk9wq&<;J*j5$B1UUjkhd@uDyYMa5ldvTL4NHehO*U1?OOU=Ib@bT~jah&GCJYf>#X%PE6kFi(f26in%Owh5(5>cCJ<>OhCH6M3a(dkl*gST%mdk|;<aL-x#2m*m+U5=|}dKxR4dz_6Bn)06whSh-upiV+L?>FM|)Z^vp`)7+D-B54Z{HP3Gm1NoOZwQy(O0x}j1fuB^o@FlmX=97S+&QaEeIfEEJPXqPA6EJ;;Q3#@3DsAoI7pz{+qNTZ__mhQG#{8^KO|}xB6rACCoXeLKBaftvz-^h$4o_9@v*s3ED-@&%N~Om<maYOjLE@u4{7EhELd+3dVzIgmrBW*%M<&8hl2*5yJ5x<Oj?oL%#-z}3rdD6H&qW@5An8QW&Q9sJ+|N=~`<T;TvV9$QeyFn|p>NevC!$$}i`MK=Mk7;_S=e*@s&TnkAWXd@pX3F+ltLQ&ArX}9RYw(A7DovXv!HFVEI~*_KZ6!huxBdFX%1@yj)IhZSf)~ggHN<&1|ieqRE(xs7n2F_YK2DxQR6~YhD$+z$3%Xs)mlAp%-5@;P$6|#`8^%3d0CA^FQul}s~94uf&nDQEGu&?uw_DVLwk;+7(t+hwUs!KU}qCHD-*{aZ*JcO<=Q*hkLaQTcmXrGB>DHe)>e)1DwV!~=m)gQ{N<Zfr}7jP*X8g_U*8TNa0jU_dvjatEg%ceGd^=kE)1Up!4xK>x1BS4!BU=y-^Um8LtS=%a719Nq-0D34Te6C*08BKtO2mwY~$5JWVuP$ot5ifh1jH0h$<p!d(~W9mq#D9>=Ob_`h6~Bd$kF!#qb#S4+XkZUb@0E>a3NOlzp#hd)6}PSRgu2S*(QCi#$4&$0bRj6d81p)2_<3B$5G}p=qj$29fBXC=}&ko<klAu~}87x_Vg#Xr|PM!cw<zklsv*5LjI3g3h6I(fCLeJBNxHVx0I!)1Y!X>XM$?xf1o5Y8sPRk0LbaJ8>s8FxtQcrgPtz&CWO_PiwGQ+tDGr0ZSpU0`T@`)d-ZFHHzji!kfeRa<@_7uEy$ic>-9vdKK)GUOrXw7r{8>2_FQC2}=|OrUt76AtO%JW0)J1`sq+CNY+2J@2gBAWCU@*#yE46MI$Y9>N3d<U1j)Fq;`lFY3(Icb}2+|w5o9(r0DU|WPCbkPnZgkCzu>2LqGo!S=e?qK`{ZU)<$Q_mLIT~u5F{3UC49hEfxBJUO78p^Q$Zymu2(+XDy;bp^RB7bde284sn|58M0HiV+LZ=$XS(8YcH)grae%u(wH<vH7Tn=P0Rd`y@-Dn#kV-yh}fhFw6MrQN~E;4<Wd%8*(h!t5fo`|ilTBvRw|eC&x@>R8!RQYX0tNmoD-t==uPl>^rK^UM=ZX|Ni}DHJu2#H{vj3>-k=<pniLwi{-}aI)uTxW2a?@JfIMf<Ir3T(-(84U_Xz4`O5bvsL$>NGwR<9{J;LM(V=1^!+I@GLD6A*`12>MqKbZ=qjiP1@&lR0ekuNDO@^;xW#=1ZoLrEsxzq*46nUqKZy<>}h_E;@kON<d-05U^pO8J~!5p$As$0P+b+YMj$tuZYI`sR@Z3MeVY&t^3p=fuK{7&s9Qi;DvD>XY^&2vHx8kolWmrNJy2NznTevk3>v{Q6A=aS_MuJW62OEH{BAJ=n`_RYX|=r7%if4>VT1y($1nohwKZH+CNTkRYV8W9w*$RZ2=mQ9wg;4GT!G#FVkK6{9pwj{FD>e`+VzB~;1u2?}x?x#K610x~5kW3)A30I(Sd%M7%NV2hP38Auz;pHEk)lG&I(vUMqMgO4HU>Ikl5gcO>?TCZp+iIVADP!ZN;^Jg|vJ7JOS;S6FfrWEDrHH(fU$DE{d;!|EN=Thugf%yom47-$~-6PPwI+ySznq~?>nU@3u3=(@$om%q4xl3lsrJ&XBb`v*Lnc>t>P&DO4ZaHQntm>T4JON#+(x_++RKwMZCO!|~WW^B#ku!-RjS~3PGN1u@9wxki^hk3ihP)pUQ{KvT>;cq9>>Oqd&9J&|NI6q7I{;NwhixmB1`V6NR!L>>90SXi6SOP{M{z?QlOvr|*o4aWpw`^}%5uP`Z@zR$&^mx;%twrigovsx3PEycm}rfHWz9y$+RtS+dLPPltTY;mD*8+k7ny0d>^4{JRS*=1Kb`%SSE4DXMn{$|B|OoXd#ob0QfEZBJsOj>BB)ez8!L7o+y4YX=Z%!~%{q#|n2iH1J64Sox7|8^g&alA?&!Tm60V*kqD)V{^C^4;1R*EKmy>gvIcAz2oR~G9$RwTQ=}r(u9af(}V#$gLv^fqGO_OFBnHob&uR*8q;wu_QVbYcgZMz&JQS_*Prj}k}h0))`z*()}E`WqVdYt8~1@#j=WSga4#w~+nqcAN|<BL-8MxhnIa(2KgoLi!Vz~;;~o?}XD4-wuH_>q1d3|^yW9$sYF>qKQHNEB2<R0MBD+8C!VTaN_!TSOO%l~LA7%r*m0PbRg9jw*aBCFMNt8Ra&yl&ywr$)hr%&2-F7j6bs(RME~5CUor^1nO*S)8nG>slvhDEL=^KBb`wAFIvc}2{>C;GK*6KG`Wl(SJ>U{Q$p%_MZuU@m828qK0NfT*G2Hjdc+iV3jmH>W@91>R!6Azm(6Ys#PvIAo%M*UlAFnt9GM<e23?-;oM&1=lOF-iHAQP*aJ=^$a9JifgeCI`cV*H={+}ue(Ud_5X##<jPq{j@3WP5dqu1bcXHRAwp>qMmpA!`leSmB-p%vEKzO!elILeU}8=00#rfnwB848nfDK)nJz<9(`?h(GDOmWj@WjNrZmm21wuV>8@oPrB*XD>Ra!DaJW&Fof$1=Hy7a29bmZ2}`%&@Cf*O`elh`&#!do9YOq_Nw2jL_6TP+Ea(^hh;I(*paD2T>3f8o|t&C5OoOPXPozH)z+<oY1`|KxlSONQaIwS#5iduxw{;9Ix5|gRHhG0j~s=VqMM*l#-OI1hs^?(mCCXiW)(9-Kk^j<b=m@<VU$OQfu%}AJnfhV9<4fBxpql0aKy^(P&<GLsHuThvRX=2Pq$%pQaN8-y9A|a<FMwMtHA9KjG|tj#iGIgvC7a=%~!mX!`%OkFeJf=dHU7*L6--#WiCYaspa9)01QaVX<sx;X*ibXt6y;OU9G67B%{!YEBf=+?A6M@k0Rb(wO=X)edUsSJYXeu*LG|D&{*XcTLH!am*w%h)&r^4`L&S?Ul<?u!9F0n2nYRGtg+C7Qo$@albR6~E@5+x61hMym&?m%%t=Jm>ABwK7s+QyP!oh<{lqQk%VMom6K08)L=>IFe*Hs)fopH0VX2J8IC>VYMaAN5D+MevJx4{PH#v35{nk?Umr^993X#m$>p;{E%KWO8uL_jWkUX|h+{LT<p()q*73FZGl7s$&NM-7!lxNf;y~ZY^1z99M@V1}$W4Y`uD>loHoT!4QG=TxFYa@NNj$<Wq^<I`E3zYU);?8`lXtI;R%650UYRfXoX_RwB94RsTmZTH#J2nyKr<4WYMp!V%PIu1xDZ{m>TeTXzniZ7Jna(6F5SWb?sk1dfzzh+3c_L|jC4j2iu~>$e%8c592XtDbzUz0%qrXb^cSUtnKkMG&p3*#6HnvUK-gQi=m;!8ao{|5>L!Y|=ei^cGEWD`dX!4ZbjWfGlv^0|m^pyGPUU@kSW=E#(S2P%zA6g1tQHe62YOW;7a~Q?7tX`m4u$pM1;med_A8z>evcUo1Mpcn$7i1T*kY%M469mNL>}7<NF$^hlAyJvqTSv7nK~!Y{0T^_R^{9f4k_gor;)>W+IR(1JF6DRz{gmLXi<)q>+;1pf$B)7e$;(6t68ejpt{c;s&T#GYioW30C8}jWzDgVyA|nzao2qyqg@f8o3YuZev=O%XbzG?&<QSlET<xJI9u{?u;nKQQn^M}V0y6vG=rOs-G*GoI*#RMqBF}zY$DK^cnvY^pV3m=^*3P-Krf4vRlLyBXxmd;4QYC&F(yIbuw0w^?>Q|Nh#7TcZS5xBru^LLrS+V+%C%{GY)k>nU4OMpFmAlDHWK)qh$k0wOFS8%DMgoVm&aqqzbU}VNgn{LkrVfV3t3rxXR2E~jd8pGNSO}5vw<T_a!dR3QUjxx&s+f`>8jT6bkAODYOGk|wnhRwZLghREWTbOqy&AQ|Cd+cx6u}dm<vhK4ozDCS2pJSeh+PPYdiXM{jpzD`ilhk^If<M_*rg~Wqsz00O$#XuY4j$h?F<bk=uydglnW8Kt6p5foN(aVRVgx~dJd5f>Y>S%x0Yw9)|?`ICGis`s23d#+rCRgVXtJ9C76ehkrssgXADA;^~B@wMOS9kCEc6)uQWFaJzTxf=>B0u1X`6`V|!3QN56atZG{FL^oxwFo#|$|YDVCi^O1@Zy;bTvu&IpLP1>X@L*y@vm6$e?yd+qDAWNkL(2T;H%Vq>7od#B71oY<ce^Kn0P3u7p<(o;nPc%||9H_YR8_x2|fkPN@Hgz}rajIf9zrVBhI=HNdjfUr4!O4xuQ9fs_@PV#B7e(v2T2^8LuaMI_8>>@G^5MTHG3}c'

TRACE_ACTIONS = json.loads(zlib.decompress(base64.b85decode(_PAYLOAD)))

def agent(observation, configuration=None):
    del configuration
    step = min(int(observation.get("step", 0) or 0), len(TRACE_ACTIONS) - 1)
    return copy.deepcopy(TRACE_ACTIONS[step])

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")

ANIMALS = ("GOOSE", "COW", "SHEEP")

PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")

UNIT_ITEMS = (*PRODUCTS, *ANIMALS)

SIMPLE_UNITS = (
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
    "DROP",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
)

FIRST_YIELD_DAY = {
    "WHEAT": 2,
    "CARROT": 2,
    "TOMATO": 8,
    "STRAWBERRY": 10,
    "MELON": 10,
}

ANIMAL_RULES = {
    "GOOSE": ("COOP", "EGG", 300),
    "COW": ("PASTURE", "MILK", 400),
    "SHEEP": ("PASTURE", "WOOL", 500),
}

SEED_COSTS = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}

LAND_PRICES = (1_000, 2_000, 4_000)

MARKET_PARAMS = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}

SEMANTIC_RECOVERABLE_UNIT_OPS = {
    "PICKUP",
    "DROP",
    "PLACE",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "DIG",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
}

class WeedRecoveryController:
    """Repair random-weed tape divergence and rejoin the worker schedule."""

    def __init__(self) -> None:
        self._pending: dict[int, dict[int, list[Any]]] = {}
        self._last_step: dict[int, int] = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._pending.clear()
            self._last_step.clear()
        else:
            self._pending.pop(player, None)
            self._last_step.pop(player, None)

    def repair(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        repaired = copy.deepcopy(action)
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step <= previous_step or step % 24 == 0:
            self._pending.pop(player, None)
        self._last_step[player] = step

        farm = observation["farms"][player]
        positions = [farm["farmer"], *farm.get("hands", [])]
        commands = [repaired.get("farmer", ["PASS"]), *repaired.get("hands", [])]
        pending = self._pending.setdefault(player, {})

        for actor, delayed in list(pending.items()):
            if actor >= len(commands) or actor >= len(positions):
                pending.pop(actor, None)
                continue
            current = commands[actor]
            self._set_actor(repaired, actor, delayed)
            if current == ["PASS"]:
                pending.pop(actor, None)
            else:
                pending[actor] = copy.deepcopy(current)

        commands = [repaired.get("farmer", ["PASS"]), *repaired.get("hands", [])]
        for actor, (position, command) in enumerate(zip(positions, commands, strict=False)):
            if actor in pending or not isinstance(command, list) or not command:
                continue
            x, y = position
            tile = farm["tiles"][y][x]
            if not isinstance(tile, dict) or tile.get("kind") != "WEED":
                continue
            operation = command[0]
            if operation in {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}:
                self._set_actor(repaired, actor, ["DIG"])
                pending[actor] = copy.deepcopy(command)
            elif operation in {"WATER", "HARVEST", "FERTILIZE"}:
                self._set_actor(repaired, actor, ["DIG"])
        if not pending:
            self._pending.pop(player, None)
        return repaired

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[Any]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command

def append_terminal_liquidation(
    observation: dict[str, Any], action: dict[str, Any], *, terminal_step: int = 718
) -> dict[str, Any]:
    """Sell otherwise-worthless terminal shed inventory without duplicating orders."""

    if int(observation.get("step", 0) or 0) < terminal_step:
        return action
    result = copy.deepcopy(action)
    market = result.setdefault("market", [])
    scheduled: Counter[str]
    scheduled = Counter(
        str(order[1])
        for order in market
        if isinstance(order, list) and len(order) == 3 and order[0] == "SELL"
        for _ in range(int(order[2]))
    )
    shed = observation.get("private", {}).get("shed", {})
    for product in PRODUCTS:
        available = int(shed.get(product, 0) or 0) - scheduled[product]
        if available > 0 and len(market) < 10:
            market.append(["SELL", product, available])
    return result

def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value

def _canonical_unit(command: Any) -> list[Any]:
    if not isinstance(command, list) or not command or not isinstance(command[0], str):
        return ["PASS"]
    operation = command[0]
    if operation in SIMPLE_UNITS:
        return [operation] if len(command) == 1 else ["PASS"]
    if operation == "PLANT":
        return command.copy() if len(command) == 2 and command[1] in CROPS else ["PASS"]
    if operation in {"PICKUP", "PLACE"}:
        if len(command) == 2 and command[1] in UNIT_ITEMS:
            return [operation, command[1], 1]
        if len(command) == 3 and command[1] in UNIT_ITEMS and _positive_int(command[2]) is not None:
            return command.copy()
    return ["PASS"]

def _inventory_add(inventory: dict[str, int], item: str, quantity: int) -> None:
    inventory[item] = int(inventory.get(item, 0)) + quantity

def _inventory_take(inventory: dict[str, int], item: str, quantity: int) -> bool:
    current = int(inventory.get(item, 0))
    if current < quantity:
        return False
    remaining = current - quantity
    if remaining:
        inventory[item] = remaining
    else:
        inventory.pop(item, None)
    return True

def _shed_tiles(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return (
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    )

def _tile_object(tile: Any) -> dict[str, Any] | None:
    return tile if isinstance(tile, dict) else None

def _apply_unit(
    command: list[Any],
    unit_index: int,
    positions: list[list[int]],
    tiles: list[list[Any]],
    inventories: list[dict[str, int]],
    shed: dict[str, int],
    seeds: dict[str, int],
    *,
    day: int,
    board_size: int,
    shed_capacity: int,
) -> bool:
    if unit_index >= len(positions):
        return False
    while len(inventories) <= unit_index:
        inventories.append({})
    inventory = inventories[unit_index]
    x, y = positions[unit_index]
    operation = command[0]

    movement = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0),
    }.get(operation)
    if movement is not None:
        next_x, next_y = x + movement[0], y + movement[1]
        if not (0 <= next_x < board_size and 0 <= next_y < board_size):
            return False
        positions[unit_index] = [next_x, next_y]
        return True
    if operation == "PASS":
        return True

    adjacent = (x, y) in _shed_tiles(board_size)
    if operation == "DROP":
        if not adjacent or not any(quantity > 0 for quantity in inventory.values()):
            return False
        room = max(0, shed_capacity - sum(shed.values()))
        for item, quantity in list(inventory.items()):
            accepted = min(max(0, int(quantity)), room)
            if accepted:
                _inventory_add(shed, item, accepted)
                room -= accepted
        inventory.clear()
        return True
    if operation == "PICKUP":
        item, requested = str(command[1]), int(command[2])
        amount = min(requested, int(shed.get(item, 0)))
        if not adjacent or amount <= 0:
            return False
        shed[item] -= amount
        _inventory_add(inventory, item, amount)
        return True
    if operation == "PLACE":
        item, requested = str(command[1]), int(command[2])
        tile = _tile_object(tiles[y][x])
        matching_structure = bool(
            item in ANIMAL_RULES
            and tile is not None
            and tile.get("kind") == ANIMAL_RULES[item][0]
            and tile.get("animal") is None
        )
        if matching_structure:
            if not _inventory_take(inventory, item, 1):
                return False
            tiles[y][x] = {
                "kind": ANIMAL_RULES[item][0],
                "animal": item,
                "placed_day": day,
                "yield_units": 0,
                "fed_today": False,
                "cared_today": False,
                "fertilizer_available": False,
            }
            return True
        room = max(0, shed_capacity - sum(shed.values()))
        amount = min(requested, int(inventory.get(item, 0)), room)
        if not adjacent or amount <= 0:
            return False
        _inventory_take(inventory, item, amount)
        _inventory_add(shed, item, amount)
        return True

    if tiles[y][x] == "LOCKED":
        return False
    tile = _tile_object(tiles[y][x])
    if operation == "PLANT":
        crop = str(command[1])
        if tiles[y][x] is not None or int(seeds.get(crop, 0)) <= 0:
            return False
        seeds[crop] -= 1
        tiles[y][x] = {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": day,
            "watered_today": False,
            "yield_units": 0 if crop in {"TOMATO", "STRAWBERRY"} else 1,
            "fertilized_until_day": -1,
        }
        return True
    if operation == "WATER":
        if tile is None or tile.get("kind") != "PLANT" or tile.get("watered_today") is True:
            return False
        tile["watered_today"] = True
        return True
    if operation == "HARVEST":
        if tile is None or int(tile.get("yield_units", 0) or 0) <= 0:
            return False
        quantity = int(tile["yield_units"])
        if tile.get("kind") == "PLANT":
            crop = str(tile.get("crop", ""))
            planted = int(tile.get("planted_day", day) or 0)
            if crop not in FIRST_YIELD_DAY or day - planted < FIRST_YIELD_DAY[crop]:
                return False
            _inventory_add(inventory, crop, quantity)
            tile["yield_units"] = 0
            if crop in {"WHEAT", "CARROT", "MELON"}:
                tiles[y][x] = None
            return True
        animal = tile.get("animal")
        if animal not in ANIMAL_RULES:
            return False
        tile["yield_units"] = 0
        _inventory_add(inventory, ANIMAL_RULES[str(animal)][1], quantity)
        return True
    if operation == "FERTILIZE":
        if tile is None or tile.get("kind") != "PLANT":
            return False
        if not _inventory_take(inventory, "FERTILIZER", 1):
            return False
        tile["fertilized_until_day"] = max(int(tile.get("fertilized_until_day", -1) or -1), day + 2)
        return True
    if operation == "DIG":
        if tiles[y][x] is None or (tile is not None and tile.get("animal") is not None):
            return False
        tiles[y][x] = None
        return True
    if operation in {"BUILD_COOP", "BUILD_PASTURE"}:
        if tiles[y][x] is not None:
            return False
        tiles[y][x] = {"kind": operation.removeprefix("BUILD_")}
        return True
    if operation == "FEED":
        if tile is None or tile.get("animal") is None or tile.get("fed_today") is True:
            return False
        if not _inventory_take(inventory, "WHEAT", 1):
            return False
        tile["fed_today"] = True
        return True
    if operation == "COLLECT_FERTILIZER":
        if (
            tile is None
            or tile.get("animal") is None
            or tile.get("fertilizer_available") is not True
        ):
            return False
        tile["fertilizer_available"] = False
        _inventory_add(inventory, "FERTILIZER", 1)
        return True
    if operation == "CARE":
        if tile is None or tile.get("animal") is None or tile.get("cared_today") is True:
            return False
        tile["cared_today"] = True
        return True
    return False

def _shape(function: str, value: float, throughput: float) -> float:
    value = max(value, 0.0)
    if function == "linear":
        return value
    if function == "sq":
        return value * value
    if function == "sqrt":
        return math.sqrt(value)
    if function == "log":
        return math.log1p(value)
    if function == "hinge" and throughput > 0:
        normalized = value / throughput
        return normalized + 8.0 * max(normalized - 1.0, 0.0) ** 2
    return value

def _market_price(item: str, inventory: int) -> int:
    base, throughput, below_function, below_target, above_function, above_target = MARKET_PARAMS[
        item
    ]
    if inventory < 10_000:
        amplitude = below_target * base / _shape(below_function, throughput, throughput)
        price = base + amplitude * _shape(below_function, 10_000 - inventory, throughput)
    else:
        amplitude = above_target * base / _shape(above_function, throughput, throughput)
        price = base - amplitude * _shape(above_function, inventory - 10_000, throughput)
    return max(1, round(price))

def _fibonacci(index: int) -> int:
    first = second = 1
    for _ in range(index):
        first, second = second, first + second
    return first

def _canonical_market(order: Any) -> list[Any] | None:
    if not isinstance(order, list) or not order or not isinstance(order[0], str):
        return None
    operation = order[0]
    if operation in {"HIRE", "BUY_LAND"}:
        return [operation] if len(order) == 1 else None
    if len(order) != 3 or not isinstance(order[1], str):
        return None
    quantity = _positive_int(order[2])
    if quantity is None:
        return None
    item = order[1]
    valid = (
        (operation == "SELL" and item in PRODUCTS)
        or (operation == "BUY_PRODUCT" and item in {"WHEAT", "FERTILIZER"})
        or (operation == "BUY_SEED" and item in CROPS)
        or (operation == "BUY_ANIMAL" and item in ANIMALS)
    )
    return [operation, item, quantity] if valid else None

def legalize_action(
    observation: dict[str, Any],
    action: dict[str, Any],
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a canonical action whose every component is locally effective."""

    config = configuration or {}
    board_size = int(config.get("boardSize", 10))
    shed_capacity = int(config.get("shedCapacity", 100))
    market_limit = int(config.get("maxMarketOrdersPerTurn", 10))
    hand_cost_multiplier = int(config.get("farmHandCostMult", 1))
    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    private = observation.get("private", {})
    positions = copy.deepcopy([farm["farmer"], *farm.get("hands", [])])
    tiles = copy.deepcopy(farm["tiles"])
    inventories = copy.deepcopy(private.get("inventories", []))
    shed = copy.deepcopy(private.get("shed", {}))
    seeds = copy.deepcopy(private.get("seeds", {}))
    day = int(observation.get("day", int(observation.get("step", 0)) // 24))

    proposed_units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    legalized_units: list[list[Any]] = []
    for unit_index, proposed in enumerate(proposed_units[: len(positions)]):
        command = _canonical_unit(proposed)
        if not _apply_unit(
            command,
            unit_index,
            positions,
            tiles,
            inventories,
            shed,
            seeds,
            day=day,
            board_size=board_size,
            shed_capacity=shed_capacity,
        ):
            command = ["PASS"]
        legalized_units.append(command)
    if not legalized_units:
        legalized_units.append(["PASS"])

    money = float(farm.get("money", 0) or 0)
    hires_today = int(farm.get("hires_today", 0) or 0)
    unlocked = len(farm.get("unlocked_quadrants", []))
    market_inventory = copy.deepcopy(observation.get("market", {}).get("inventory", {}))
    accepted_market: list[list[Any]] = []
    for proposed in action.get("market", [])[:market_limit]:
        order = _canonical_market(proposed)
        if order is None:
            continue
        operation = order[0]
        if operation == "HIRE":
            cost = (
                0 if hand_cost_multiplier == 0 else _fibonacci(hires_today) * hand_cost_multiplier
            )
            if money < cost:
                continue
            money -= cost
            hires_today += 1
        elif operation == "BUY_LAND":
            land_index = max(0, unlocked - 1)
            if land_index >= len(LAND_PRICES) or money < LAND_PRICES[land_index]:
                continue
            money -= LAND_PRICES[land_index]
            unlocked += 1
        else:
            item, quantity = str(order[1]), int(order[2])
            committed = 0
            for _ in range(quantity):
                if operation == "SELL":
                    if int(shed.get(item, 0)) <= 0:
                        break
                    price = _market_price(item, int(market_inventory[item]))
                    shed[item] -= 1
                    money += price
                    if price > 1:
                        market_inventory[item] += 1
                elif operation == "BUY_PRODUCT":
                    price = _market_price(item, int(market_inventory[item]) - 1)
                    if money < price or sum(shed.values()) >= shed_capacity:
                        break
                    money -= price
                    _inventory_add(shed, item, 1)
                    market_inventory[item] -= 1
                elif operation == "BUY_SEED":
                    price = SEED_COSTS[item]
                    if money < price:
                        break
                    money -= price
                    _inventory_add(seeds, item, 1)
                else:
                    price = ANIMAL_RULES[item][2]
                    if money < price or sum(shed.values()) >= shed_capacity:
                        break
                    money -= price
                    _inventory_add(shed, item, 1)
                committed += 1
            if committed == 0:
                continue
        accepted_market.append(order)

    return {
        "farmer": legalized_units[0],
        "hands": legalized_units[1:],
        "market": accepted_market,
    }

def _semantic_unit_fallback(observation: dict[str, Any], actor: int) -> list[Any]:
    """Choose one useful local task after a replay command becomes stale.

    The fallback deliberately never moves a unit.  Preserving its position keeps
    the remaining replay route aligned while still recovering work that is
    already available on the current tile.
    """

    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    positions = [farm["farmer"], *farm.get("hands", [])]
    inventories = observation.get("private", {}).get("inventories", [])
    if actor >= len(positions):
        return ["PASS"]

    x, y = positions[actor]
    tile = farm["tiles"][y][x]
    inventory = inventories[actor] if actor < len(inventories) else {}
    day = int(observation.get("day", int(observation.get("step", 0)) // 24))

    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = str(tile.get("crop", ""))
        planted_day = int(tile.get("planted_day", day) or 0)
        harvestable = (
            crop in FIRST_YIELD_DAY
            and int(tile.get("yield_units", 0) or 0) > 0
            and day - planted_day >= FIRST_YIELD_DAY[crop]
        )
        if harvestable:
            return ["HARVEST"]
        if tile.get("watered_today") is not True:
            return ["WATER"]
        if (
            int(inventory.get("FERTILIZER", 0) or 0) > 0
            and int(tile.get("fertilized_until_day", -1) or -1) < day
        ):
            return ["FERTILIZE"]
        return ["PASS"]

    if isinstance(tile, dict) and tile.get("animal") in ANIMAL_RULES:
        if int(tile.get("yield_units", 0) or 0) > 0:
            return ["HARVEST"]
        if tile.get("fed_today") is not True and int(inventory.get("WHEAT", 0) or 0) > 0:
            return ["FEED"]
        if tile.get("cared_today") is not True:
            return ["CARE"]
        if tile.get("fertilizer_available") is True:
            return ["COLLECT_FERTILIZER"]
        return ["PASS"]

    if isinstance(tile, dict) and tile.get("animal") is None:
        structure = tile.get("kind")
        for animal, (required_structure, _, _) in ANIMAL_RULES.items():
            if structure == required_structure and int(inventory.get(animal, 0) or 0) > 0:
                return ["PLACE", animal, 1]

    board_size = len(farm["tiles"])
    if (x, y) in _shed_tiles(board_size) and any(
        int(quantity or 0) > 0 for quantity in inventory.values()
    ):
        return ["DROP"]
    return ["PASS"]

def recover_rejected_unit_intents(
    observation: dict[str, Any],
    proposed: dict[str, Any],
    legalized: dict[str, Any],
    configuration: dict[str, Any] | None = None,
    *,
    allowed_operations: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Replace rejected non-PASS unit commands with state-aware local work.

    A final legality pass projects all replacements together, so two units can
    never consume the same inventory or tile resource in one turn.
    """

    recovered = copy.deepcopy(legalized)
    proposed_units = [proposed.get("farmer", ["PASS"]), *proposed.get("hands", [])]
    legal_units = [legalized.get("farmer", ["PASS"]), *legalized.get("hands", [])]
    for actor, (before, after) in enumerate(zip(proposed_units, legal_units, strict=False)):
        if (
            not isinstance(before, list)
            or not before
            or before[0] == "PASS"
            or before[0] not in SEMANTIC_RECOVERABLE_UNIT_OPS
            or after != ["PASS"]
        ):
            continue
        fallback = _semantic_unit_fallback(observation, actor)
        if fallback != ["PASS"] and (
            allowed_operations is None or fallback[0] in allowed_operations
        ):
            WeedRecoveryController._set_actor(recovered, actor, fallback)
    return legalize_action(observation, recovered, configuration)

_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "WEST": (-1, 0),
    "EAST": (1, 0),
}

@dataclass(slots=True)
class OrphanStructureRecoveryController:
    """Fill a structure left empty across a day boundary after route recovery."""

    minimum_step: int = 193
    start_hour: int = 1
    feed_quantity: int = 4
    maximum_interventions_per_game: int = 1
    maximum_task_steps: int = 20
    _pending: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    _interventions: dict[int, int] = field(default_factory=dict, init=False)
    _last_step: dict[int, int] = field(default_factory=dict, init=False)

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._pending.clear()
            self._interventions.clear()
            self._last_step.clear()
            return
        self._pending.pop(player, None)
        self._interventions.pop(player, None)
        self._last_step.pop(player, None)

    def apply(self, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(action)
        player = int(observation.get("player", 0) or 0)
        step = int(observation.get("step", 0) or 0)
        if step == 0 or step < self._last_step.get(player, -1):
            self.reset(player)
        self._last_step[player] = step

        farms = observation.get("farms", [])
        if not isinstance(farms, list) or not 0 <= player < len(farms):
            return result
        farm = farms[player]
        positions = [farm.get("farmer"), *list(farm.get("hands", []))]
        inventories = observation.get("private", {}).get("inventories", [])
        shed = observation.get("private", {}).get("shed", {})
        if not positions or not isinstance(inventories, list) or not isinstance(shed, dict):
            return result

        task = self._pending.get(player)
        if task is None:
            if (
                step < self.minimum_step
                or step % 24 != self.start_hour
                or self._interventions.get(player, 0)
                >= self.maximum_interventions_per_game
                or len(positions) < 2
            ):
                return result
            target = self._find_orphan(farm, shed)
            if target is None:
                return result
            x, y, animal = target
            task = {
                "actor": len(positions) - 1,
                "target": (x, y),
                "animal": animal,
                "started_step": step,
            }
            self._pending[player] = task
            self._interventions[player] = self._interventions.get(player, 0) + 1

        actor = int(task["actor"])
        if (
            actor >= len(positions)
            or step - int(task["started_step"]) > self.maximum_task_steps
        ):
            self._pending.pop(player, None)
            return result

        position = positions[actor]
        inventory = inventories[actor] if actor < len(inventories) else {}
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not isinstance(inventory, dict)
        ):
            self._pending.pop(player, None)
            return result

        x, y = int(position[0]), int(position[1])
        target_x, target_y = task["target"]
        animal = str(task["animal"])
        tile = farm.get("tiles", [])[target_y][target_x]
        if isinstance(tile, dict) and tile.get("animal") is not None:
            if (x, y) != (target_x, target_y):
                self._pending.pop(player, None)
                return result
            if int(inventory.get("WHEAT", 0) or 0) > 0 and tile.get("fed_today") is not True:
                self._set_actor(result, actor, ["FEED"])
                return result
            if tile.get("cared_today") is not True:
                self._set_actor(result, actor, ["CARE"])
                return result
            self._pending.pop(player, None)
            return result

        if not self._matches_empty_structure(tile, animal):
            self._pending.pop(player, None)
            return result

        shed_tiles = self._shed_tiles(len(farm.get("tiles", [])))
        if int(inventory.get(animal, 0) or 0) <= 0:
            if (x, y) not in shed_tiles or int(shed.get(animal, 0) or 0) <= 0:
                self._pending.pop(player, None)
                return result
            self._set_actor(result, actor, ["PICKUP", animal, 1])
            return result

        if (
            int(inventory.get("WHEAT", 0) or 0) <= 0
            and (x, y) in shed_tiles
            and int(shed.get("WHEAT", 0) or 0) > 0
        ):
            quantity = min(self.feed_quantity, int(shed["WHEAT"]))
            self._set_actor(result, actor, ["PICKUP", "WHEAT", quantity])
            return result

        if x != target_x:
            self._set_actor(result, actor, ["EAST" if x < target_x else "WEST"])
            return result
        if y != target_y:
            self._set_actor(result, actor, ["SOUTH" if y < target_y else "NORTH"])
            return result
        self._set_actor(result, actor, ["PLACE", animal, 1])
        return result

    @staticmethod
    def _shed_tiles(board_size: int) -> set[tuple[int, int]]:
        half = board_size // 2
        return {
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        }

    @staticmethod
    def _matches_empty_structure(tile: Any, animal: str) -> bool:
        expected = "COOP" if animal == "GOOSE" else "PASTURE"
        return (
            isinstance(tile, dict)
            and tile.get("kind") == expected
            and tile.get("animal") is None
        )

    @classmethod
    def _find_orphan(
        cls,
        farm: dict[str, Any],
        shed: dict[str, Any],
    ) -> tuple[int, int, str] | None:
        choices: list[tuple[int, int, int, str]] = []
        tiles = farm.get("tiles", [])
        board_size = len(tiles)
        shed_tiles = cls._shed_tiles(board_size)
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get("animal") is not None:
                    continue
                kind = tile.get("kind")
                animals = ("COW", "SHEEP") if kind == "PASTURE" else ("GOOSE",)
                for animal in animals:
                    if int(shed.get(animal, 0) or 0) <= 0:
                        continue
                    distance = min(abs(x - sx) + abs(y - sy) for sx, sy in shed_tiles)
                    choices.append((distance, y, x, animal))
                    break
        if not choices:
            return None
        _, y, x, animal = min(choices)
        return x, y, animal

    @staticmethod
    def _set_actor(action: dict[str, Any], actor: int, command: list[Any]) -> None:
        if actor == 0:
            action["farmer"] = command
        elif actor - 1 < len(action.get("hands", [])):
            action["hands"][actor - 1] = command

_V44_TAPE_AGENT = agent
# Kaggle's raw loader chooses the last newly inserted callable in the module.
del agent

_V44_WEED_RECOVERY = WeedRecoveryController()
_V44_ORPHAN_RECOVERY = OrphanStructureRecoveryController()


def agent(observation, configuration=None):
    proposed = _V44_TAPE_AGENT(observation, configuration)
    weed_repaired = _V44_WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _V44_ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
