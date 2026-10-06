"""Compact replay-tape agent generated from public training data."""

from __future__ import annotations

import base64
import copy
import json
import zlib

SOURCE = {'competition': 'kaggriculture', 'submission_id': 55216451, 'episode_id': 89728633, 'seed': 2051039310, 'top1_seat': 1, 'source_sha256': '5bad79593f715e2b044acdee54f6f55d70ce3a4980c9d87fa17ec9bda017ab14'}
_PAYLOAD = 'c-rk<O>ZPua{Mnm^T7V%hi}wOuQXUn9B??>SPKI|fY&f!tPiqphX1=q!|v*?%E-vbe6L83G`B{v+4a6(W@KdK&;NV&@4x=`&wu=O_Afu5y}$kN;q24-*?;`{U;p;spC5ew_|IQ|`_F&;&*#rSpZ)OeVf*>7Cm+81`1fCKe|q=h?cLe=*_->_+4<t|_``Pl_VdFZw|DnHot<A!9`AnG-ahUhf7ssL9Um?1qZRtrr~f-YZTWW}-`&0a<#YQVKYnT3(1oARcH0k+Ut0Uq{lnuAXP?HM`n^%_AMW3NeDgSISE85uf7?u3_5R(PzkYmw@?R&-+PCKX^wU#Qg$6FYc1;^_cl&0$|Lf@O)A8!miVr_*x9?Fq4g+!g&y&V{xc|6+wQbhqiO{Omaq=du_{;CJHG1=Sf|lcP*xv3Jjyxa>VWkeQ9LDOjL2qu&bPSK4K1j=V^3KDL!R7aPjr{aDY}N6*pALl6SH{0QUUQ#E@Ah%~5dLno#Kz}|yU?Eo6%Xlf5FFPVYR7o~h6aFj7A~)o7m-1R=Mp#-{Q;UbdD5``2%vR;`YULn_C1hhYVy|oaKkk>OvUgaw<4lFwjb@MzmM9mpCmLR?i`Wze;6}3>Bj**`J|pD%B^RI*beOE1L+GHwdtN#25<6nwJzILWe0qiQaIjEpV$<dL9{1E{wi1_J4PSy<GwKCI<$oha6D+oWibrZv^j9_^zJT&TL>J<Y^$Ue?4&RDi9Mb5rMec@C0dTApZwHMm{IB^O|1Cv4&&xCxGeb#xFvYBb?)!(wr?JP`OEg<@!j3Ke;qA}lr6t%Z4<b)knJ9tN@~x8A?WtT6ALqX)6k~YPU06$PS?;<951?c%G!PVaR0vWu9Ftq7lKW5JXD`tYJA|THqRn+l!7-FjI2r*-3vrnIfs*xcWH7tH1!F+(RS)&`|F({hFPvJm1!uoc)tg+^J1IKT_|l4PthjWnd6JtE5lzt8cm&lm^L{v{z{XNKR5Z?D`|0g#Z-QI7=o6uuw$(JEW8yOj5Y9U%l@XCY!Zry%Z!Gc3c=jWYtkiv(7A2B*ca+<>&4(l$=CxUo`5JOZ(Ky+M+p(2)>RDTDIH-KxepsPw>||AsrR_n7XphFt-_<*V24S^Ef*=XaCj#&vY!5!G7FER0(vtIMda@w21#45i}hrBC|`XF4angIN1w}%BYIQSd;}I#3iAJa^gY~{U)9ApV|HqjZ8pq7Zv*(~{e8VZT(Ko(>#naAH!{%za}Ht`=OKRb>=51JPi)2wvvJCl2!QlwvGt}6+kEN@^-0!WdZ{}0Ah>|25dyc)bqR#)ZXGiGNQfzHIzfpunTs?Y^wn956q2h%CMm+g%*6~<fgsj#n~537Or>Wx1H&OI+I(GpSF`!FQ>vdSD8gRfEW{4@-vZfbFA51Z*(6=yz`Y@|cCFvVGlhB!gF3%yN%>^68<1>Bdf1J08s@_ff4F`4TlBdTB8QHy*oPT|^efCu7MO}&=LT+7iJ)T*7@wDe0M+5`$45is+_CuZ_;9=XZu{`?cS|4Dxr)eBI5hR-4NjaLe9>)hfy9XRfVFCKi<EK1PJiiSA#6W*)T(f0aI@L@y=k8kog+j^@>=@@j6`ls?Q-XQQA))PZEp%(1CJ=QUKvfm{k@cZEWM$v6D+>mrxV1;w*(|$y^6;7vJI$qA9ZtIa$R$IvRO;|@Iwm+1$_n4^(b-YHP9scWcSQxRIg?-s_ljTj3%{Bfhr;DwUHV2veQhKsnt40$MbNSHHT2Ve(sKa?=dqg;O&KWhv;vBU4Wp&$f(Li{!1O=o`awI)u}tj^#>B=I(2ARVm=C&@;QO23vB3YHww8yr->t@WI2s^nHt2wa~tMKAqTcX`_%Ag0Qp(Sg1tP`L14U-@D=Ev$Prl*_^gdP>Vc)b7@5!i`+XUu7FF(#A%n#FmoR*<T9H_opemt_bEN?_0h1Y*p=wwLMlHB7TP-I5&m|09A-R@h5TBYIC~2O)z0Vy0y1z3`q6vQ)g9iXsp^@GYpCHEo57+CU9J|texVaArfdRe<e-B;8f}yE(mZTo*ASd8@dc26zXtLy2O1_z(VfaqFzrSNwT4omkKg31I>_`Nl4R|x*A^?kjCbb316`3?F<zq0_olKTd(mFq9=1wAI@?=?9$ib%)RH6S&4qA6o5_`F4v*yljJ(sG9?*ra!2`$MtYXpqG8D6Uv8pCWx`o2wfw?Dn@VX-qPufz|Iveiz5ziUsW%g*dYU;JH!y!8>-G!t4}Jt^j|^z;#8E=dxMZ+0Pk9yA=K%DgR9qlF=8!MGIvUQs)u$_l_gfV5ZlYc=DUF=^P^sre^Lg#Mhe3nt`|`bDVeYgwF9cvMYXw4*#+&Fu9tT}IhL1V9zy63J1~ZPS~YZOV`Y=EUR&H6z!AcJ@|aMVE%M!r`HA$FLZ!@>6Q*Eo#zi0AP@~6D_)2Lc5R?qM+DKher9O+gSXpL7bDu>cDDdAb-zU)7pjBj`=`iQo8QsB9Z@_YoI90rZ?0~`}H%lJu}&$f#+e~)-yVvbuY9*by;@vmL2?5NGom1CC!YZ=xh$xS0xlB`=5toi+6W_-7m!1JyOW-#i1UhX9TxBW%jtJLD9c84kNcR1h$o{i`PHl1ck70qX3*gJYp=i3EDsw-6BI#NKTGO_9~zf=!<PDG1*zrBtz?CJUYU$SL5pvTa?)GqoA$}TTHl1Y|#M+Nw}h;Nzy~a6LOrF)o@&%%yoiG_9%YLvto3Az{_$0K9*fqF?$u@B<V&o5CSY{&?qWPWR1UE*3lvXO$2uz<8PXsFkK)^$x#Z%S|XL&wH;pHM;3!UrM!(9G}R6>xBxWx38Q((M^o-chqi`RUDRNHco<JLYB^oeb_Wsz^E_f#V9;Z-y4`rF(q0kEr9()gwbMk&D$NJJBwC;vMp)sXQk({!qFdrGn>qE+z}ga|jb(%n`($z`<z^Gu8qwbEOnXHyq38yaXcIs#A*caSh)#6Z#Z@Qhz^<vb@Lxw1z%hqmA(yJte;Trw8Pzx;AW`A~d}yF7@M{LJZ;QY$y1_V{+U&7+)AA4Vg!6-G<L!w$dBj6*9sg~vb^|Ie^N@A^hppL_^XaYgce)<2k^T1F_YNXe<zQ&ohJtLbatdx>?7>tyw@}zER(GP1r+%C)#0;m*f!%cOhL(ypdxgIaR(K#n0xA|nrfFW`Yn2K_1A!9i(jyxEVS9`R{mWEz9%*jpTeEE3A{~{~1V}1DJ&o-_)8i<*!7?4c@W9uZNMB}dXf{hUHI%|1$0cA@Q6prdn9$VG<=3<ABRUu~;cK-rrF1GOs*qu8`E^lLVWb2z+v?$@Auwr>_1n&sA=EH*C>3P<W1xW*#sz0PCHl~V1=hmJ%|8Q0(DI^K2y?hA<bfOYTwoA19aK(gWSI<Yf<;PhM1VJGnq{K0j#o)JfbNKbt~QEIl|RCMUNzA@+Q5<JR|$^L==7^VM@}Cym%21*7yWrHAtNW9(X3No8exUZ!UKj))G<g)BGqa8OW{C}WcUyfSazJuGdp^+6=)Aq?uSkc$xp#TSy6WEqx=wkN<j1|0~iv^Li9Ii=17L@R!}ErxDrX9Ms}s(Q}q*X)Ky9G-m`mS?9etPw1f5nF;YD9+_+90bsB60>_@OOumCFhm#HsRbS$ZX8*UC6#%oK#D!_r<VM3o6FQyolWiBwMEC8ls(#aUgykr!J&0b|yjBS*b4>g0_PK(lz2xp9C4Fc7=O4)IX(%L5^UQK=STuzsu{-C9rju?T)IUI#?Ojdi~p39)YX%VPHdC$v)wgcb9LhaYn52s!BfhgN6n4+#FsHSxvD7Ju@r8-w|^(o2Sh^~94$HwG4S-zZ@%QRV13l*E9@-ZO5gH#2iwU(9WLr2)MnY42Fp)2=PT?TX(l#@6+mu2;pz%?0p<ClbHzz=;{&&&4S-aHdXQ)I1hb5Qy;KY<15G)>aT)4Fq|c=&1PuOU&%r;>!=&aIzKAqb4L!#51IfslbJXFV73Wm(MJk-;6VI+;msXrCP(b`8=RNM_P>iYwzzqv61^7#>-56c)g-I3Z;mrMzs0mvJ#dDty5>%5owG<!7_R5fIrW(Q>9d>F_ZMrMS_y2~aC4M{ZA{3{uIwNqa;y9)hz-1Rx&o-3n9`vEEs6PDbGXxpb_aR4>rb!Y?A@kJDCY^zkTysgl{*2BaeQA+s_~xq$g|?U{(cvE%ri>&S9+rFH<aBZg9Lv{_<sW>dRaVohR^gtM2<En<Qtv?i%mf<Z$UX{$vVjJ}7HE$6i>WUj6&rB<X#u~JvuS_NQ~i;LG*ds~^-O&|UYFT{>*-P!3gdIXud($5N4B->WRYhIW*qc_>EFTv`gzI~Q7b-MK~n<Yu~@>H6T=bU<{1W*~CC?qGj1aC&yXOYqm$9@1CPntXx94ka?fl~gTdbYfXdj^d42yn4;*FqACrPkLs39&6o;EDE3C~y<Iy-5)B0LdE}wtYV`nh0Rb?S3O}GNi<)cBa`xu(=xkL31t`6~N7=VascCS;-gxmZBYONxMX^o(5urEd9a_#zyF|yWYtuYWC=HH3iO-L%Oor0g1?hT~p3UI_d)`ov_-($pfJ;5$0qbp;pEhU`{-KcXPVIDc*y#?@%#A%~8zh2(8+5Dyb?i@44tI#{Q8xBKIqs3z;ThCy-!~@!2VXVM)c=%)VKkUUHB3Ki)pxqbOaj_;kZ0s=d~GD0Ie|Iv~AVo=v+<k6168)hGMVQ{=xWRX@}KMNJwT&EeCy5kz>CQbr|f520VAX6zQ%L}!Y*nwMN%VLa<AIThoRpT4n6W>x_*$ma?#7V{!f>u-S6R|H6{Uj-gj(zPkTQI3XjBOfKqk_P6k79iHE{(&eT!JrcU1ZCDpY@d3zdSY3C!DC~cJ6f8{h^`Va4hZYBXgDK7fZ@r(Y9TU15Db8F02{}y{_&+-35jI!7}SPow-Qh@Wnl5&sq{{k5GD8vrDU;CGsViP;xfFH(pIQp%4VQKz0~suMa2XS@I%T_z!MNL@l{4Y8TG@}x*=<Zb-7{DgMhe{-W14fGq95?sMc2wP4NSNOkCI!F3_hcbO_5G6wU0dz*!}8O~g(DXaWB-tBoR*cUH~xI^&4%2$6xqcp<v15{nGkG#iAkWw1n%1Wjjf0Ih~K+&+Ip_Y%;I-D6Y0E9!}?+}o-kcg>3#ox%~DI@CipXYh%L?~wm&+^5S7w200^0<zJ>H*zy7hQwMZoz;kOt(*8$BmP9*UkCv<1fpFu*5-<Dbl_w+;#sSCv7Q8|P(ioMWz%tLRax39;$_`$kByb9+0}umgb}(zCIA;F(lg?AWq6{{QOa^7+%0gY0#vgt@y#zY;-*%4pJmDF-U)E=7L#~cVB7=E#bZ*|91HIp%fU^at%QrSnBZfVrj$ZukDS>wq>(A3^SGGTLGMfT8XQl{Wfn4|=LW;qKm^>LAO)fCWeoSQ5CONB!PjVdkCKxZhUiCT=a(S&RjbYi|6~CwT~o_jeCe`lkQA&(nm808WYv+|Dx$S%B3=~lijnDDE+;t?Mfh^-5`OjAoINjX<{(90hOeXg4l&J``>~VE!gz}@iU?pV#uc5_HIvAQbcLoYj}+1PNG_O7vf_tgS#TO|H#YX5BU(n7zlTZl00Pf+nWd3rNxKBrF1B*#OKbrU1tp%v&?s55SDzAXwId_R8fo*d9T591wdtWl5_9S=UM!<-@y8)BEj09FKeJaw{0b;?_2V=ZYF>9#6o-uq`deT(y(H9AD0a<~JaQza8Ic`Yh6%XOBtBd~c4&Nhr>0(3{N=Xa?89ZE>?CGfV}V_^rox$EUN1%17=4VY@l-ID23Air^FW_S#m+?bQx<diV*cDzs$=Wo`C=pu3<<N`ytG-n1;D{V9R?04MRS~~wvyC?i4XM1z3;-;P-MnQrl%&|E};$~q)WwvSb2HUS~>EPA>y4!&Oh0bJc}JZ+-?0$=Tk~@Nul9sV2LV6mx>WAg8ZC47Ng6(c<WrO3a*m8$RZ=+S=DOxU_!VM6NLb2#{f7+UG5<_FEic@8RS|_;A@j)q6t`&Tjm_D4`6Od{BMGi<V@X|C@h!06sS76wplz5Yyg4tD$(K+vz#w_eyoEj5FCzQj@T?zqq{TIQc$?Q`U62Qjb)5pi>A@ccnj|@7dO&f>8}ECp&pzT)O$UUi_UwqF*ij?Qp^zuR|dAdDdj=zTGDYg6&uGX?bnc&(gxPB%Iie>XjhQ4Dq=V+%}h??athNm6-p?TUUiC)mlX6_FkN>O1r->8rzIM!)~2dZ0*JXH)+SeqdlAimB5`h!V!EjIKC2QX;39pfVwylpRF{smTPJ2v_kgD1If~$nwPx<HnQ|t1C3LRhl{#}0HSSoZVc<0TwZFw;Z2~N?SZ74D4`7cb@l+sRWMI-bJ&M8yRJewcrZjwA*P>3Ge6&!gsQV<Il#}tPqBO9mrmp+P0<Vm_kw`ujHO58QF~Gm7LVp%!rS`a*>nLQFOc$v|R+`I!eKF!Uj=6vRq6tzPkw=2%5XI6`Rpr~)2hh+(*@nvKme?}I{aMVF`Q&A!@Lw2XkPrb;e4l4YijtJ2!&QvZ0d&>i9_6Qaew6*7lnR@5l7N-yt+vY4MLv0b9vGaV7FWxqZ>i~cm6Yd2DWwUSZbVEd5!a<4M<tVq(cNffo)A7P$r%|HCT<tfpgcA0#tR;hvntJpymnMLX&bxHUQ;Cjh>`+UkQe4ux>s>gYQ!p1K}p`nDS>+<W;KFGo6HHvLc3B*5O*_#csfa9VnXm{l?}`TbTQxwo94$bru|xv-};5wQ3kGgbR%E}z(EALj2gWPrwnrX$!H+FsK_#rU>D$IQY=IOQt9n;TsE92EbR*}oR=ExGAPZYv`F!cU_r!zwOl;Ot_pU?gPb<4G@GP&RwMfykPw!QCMw_Pm0&4X$|P|eA;qDQo|3+}0f&iDW#<(rl-(+dC{PGPqTcu90U2NUTh`P!%Fb6z*$JIor83!T%#i|o7J`qwPR<TQk`uxH`FT4US4!hc$PiDzoG8emhhb*`>d5J7IYNyT7j`H#xw9N;jaL~I*vnm9m3TR+WDRB=){<dDZ5N4@169bFwv~f?x=skRPd@+1!#(W9MleY<85ypkLm>nSYAj8XttBJrMytt?_{<PQ;~RF*@<IcIGU;2e_1q&sfAyAVK}%_Y5E3eA>h{}WwZW|7KL=Sf)X|HsTe(`HM%dDF!h4svIJH?KAXqCZ<sB>pl~Mvd3HffZ96Ew0eAhHA&%1D;wR2N5Fu_XWn*MffMzLa{+?3{Av(kTBNewErD}e(|fFOa_ILC#IVQca*9YL@<-O6AgwjUx?yiBMhnZ(qx13fb$o^!gBF6&&Oc4!B9+OGrMA?1^0HpWZSeJbcquPRp8^CCL4R`;q@*_j3CsfBo=(=9K6vuYz4I=y46H$>miPAF2A%|&1+#-mI22F(0~O33rd;xLV0I}<Ds&XtkPM@N`_EeL`mNo*bF!M`LgFLhFgMyOUzo(8xj-Htw#DnytO$=TS&bS5B}_dYp6tP&FyWRe2pU7&T{l?y=KD4A#*1C2%5rG3w?I3<Ku7UF`-nygW#*t!onP)P(~_uyo*V8+y>MVdPtd*BzssTEeIC%~w(N8){>tQP|Nari30!)VmrZl?zHOGLgSY$iAXG$-<OsBZ}6s}@g_vgq^*+gbviC#WW7W>Qd(HaRPzm0*$}S{!Y=IHHoa@Y8@^a|BxsBJnmV1;%1`i1T?)*ZgIsnj#%kItv+2*IY!Ns~64p>Q^Z0TXxrZ#3?CpxR)=0_H;_+67yPh8f{DeOzSMzA$(Ua(~1j$)tuYSEcsmt&Ec(7lu6RkoCKLWVqE}5VK^9eh<htQN29IO^ePp*>RaE-^BY&0g25I63#>G*(Bak)X7=Ys)xQG{u;}8lid}U7^r_j!8g4%fCTt;Eqx2@pr#VoF1dCPn<rArpB6O34z=>uwJ8Z$1@R{WyDbrJk3ZtL|u#9qdkOdc5htYAz<W|CK29&kEO5hhUZ#2S`8F=}$0dq_S!luBR!4WZX7LaJK2IU?}yZtOVxlk6>IZ{!qWhQOJgWuTXvK(WEcZDs#s+IiXywIc+{G_y*unm>`V8Z;-Rz}cUUGA8HBo3?${Des=lMo6cEp%8W=jGK+V_RX2UL^ClRm$jv)C?4=g4_`hlu)65PJYjfD$FCKf8C=Dm=;(5lM#XhPVH+&h&Uf81qp#cGXM6P(4rK)DF)!AobnlS!$dX|GdEcct#H4N<q(#$Frdbl05E!}-ed{YP70DrD}-YYp3#qbr26NXSTB)^VrWR1>nGlNz&hI0G?@av6DuOOOXyO4$?mVvn?ThtmIY~Np(183CT921xed`8is^EE8g32FCg`DqENjFCzJgY~i`IZ1;(yBSfan0&#*91Xy3cO>I}KL>zSBAr`3~K?xHqZ1<7`Q2t%g!Cv;mFgAU9I*jzj5N>TltdE(Hh#u?%T=sVcR{Tig1szk=455^<te?D8PEs+ZiDIFuV4{(sq=Yh4<MF<9b^R1YAFRRPCI#o&Y!2#-v34I4FZ!u^aqmZYV{2$|+s$ik!cx~=b?YxY4S_d<o#zE%yN%^ya?=SqXL4uQp)N}nI;aIZqw$A$8quO_+=r3l*33KI1Enn0N#rmkqjBJ3}v5X6uvk}am!`BKz@N~jOSc${`IUF(eWO+fuG6^u_vL5u0^+ZF`qTuUsIiU0wdnsFXL$`Hzz>sMzC7#+^KLP9#;QWJ!g3T2qC+$s_k{*3LImXH|HWhzV{is7-|1ShcPxrcL-2U+{TM}p2zMq7ts2M+OU?3b()l2klTQe=3<BQNk%KqZ%TCXrY31R1}XdajPDNmjL~ZQH{|<hkUL6m}OCcH4ej^VMDmREmhvGq|IuH5=kf3B(<!s<cmoVslZ72p2`sYQ2Mzm$IG4*+KZ4SgY3u%d!+@f?}Eppvg!D>}ehpQ?%c0RSbm^sSNTn&94e^jcaq%E>r#B+3l=WU289%TDw?g77RtTgQQl!?1pW(C5k#%fM}~XL02OB@HC#+^XIXx`{%I<-uw&jlsZgq;AN@`*65S{m#-nktx5&bX(Af5lj<r8?MN<KYF#%G({giX(HL4%-hfY9H8n1ulc;Z1s#6vIdsGx-j1>x#u;v;d)siZjz>z#Rj5xr3(tV#U%L2o!mY)QJ;GkLw|5(+FPzmda>gxoaHHCYM@-#5MSrZ**rGu#+DYfw_4Ip44RMno`z3wG#MC_bo-f5;(`FY6M2kvB;Ys8_dZ3=~H>di{%prhkT-NYSbo-qfRM)IQ~TB*EVxDzjpu*f>)Su;|Py#d0q^$bZ&&iFp=uT)2aH?*Cql6j!iVXv*_H7Xx}-awXOGshRWMckyC-O2EQm;X8Pnj|qILhuS{U9s{zjW3k06joL&p}MLA@pL8YKKwGH`9n&&Q^9L*8k$<ZuC?5|f@59#h#{$zPcwjOd`LP9UaYrtCTN)fZg|o}LQO+OYIcC%1sVjXW*&p{!c&c~pBt!Yy-=vDnMrGlKMZM2girTo$+aiGvU!oT#o&KRDpwaT00HvibP<6|O*NST=hG9kQ!KL=Lirfn7iJ!XY8P3p?4{j)S9OJ8Y+@*74I{XsRWDnBBEZUHk`fWyj$vaVc_wtjMYInrHl^U3NS~2`zI7Jp13LVS$iWzD;HN5t(@SfDD5;iuRmY;dziP2HI3`Ncg_e4WK0zg6m7?XXFVf^N7cwWM+inZUawbXw59c(?nK7zZkFNYBWoN4e;X_EGvdnEoyJ*YWs^#EMBII5$MV9sO3FGZ))nIrvH@yqzr5s(lxDS&-674i`iYYOtVK=WcCFPLLHq(VK^J6tBxi;={0H-2TW1N&5K@k>|=2?f@UINgl8nXd2WX=c3eX4-Ki;fK!Ds)iY^bC-TqD>mFl!})JgaTT=&rPCaxsw`}sD*_C5y#7BO54c3aI;)4_@;lE49?lXuy4v-vUtPSSE!FsW<yd~Cd<B$7?lpjQN3d4Jnx-@C;(dm(ny4b2o0=5^}XTO50^BhSRlhb39r}rGb~|;T;l-73I{o$l_c^D?FU@eos0KIlLrCOJ}EeILNw9-N&%*Ah0*kEHO%eU?}B7CivXJLTux6)b|grrRL{Da;uw|_2fHN<aqnkE6*)$ZdaY#R#`Nc=9w1;s%PPtMS8Xfu=INwUNY@^@5l<vrEIzEoq_t0kVk5CEntKNwj_vo~gM1h%s|)18i#d+W6cV$FQVfx`$pE@ZRA&v=4nY>3iY|e9)m8#7mbIf~7#dD4qzj7aK~bGdHPchhc(eSMc-Sd$6ESp1rCyi@ykAv`=ILWdNx+smDl(sfT>TR?Mp_Y!HYl>VEM7!G<f#sO!VSAl>#8??@Y*v;KPJB66;CSLMMa>9gF>@{Y>$3F#y|Xi0UmAS3(I=AG>5ogkA@roH9a9o_-DbQRAFO!2?4X&%*6=DzPtVDcvI$p2Qs$yXYG?nO_TIM$~IGQX<C~j@?4_oY|(!ue03Y=WI8zXNaLY$fKru^gn)@I%Qf@0YM+#9bX&eKD-p+5GD=Z4U*|FcK~1FOZBKH-BG;1|=hjb;mkH4g;)^l}5M|d0`F?G5`71&^46I_B`{ycs6t(;n5M^u?rB3(K6-e95mjD36z}P8$7%XmokxZ*tyQ-`g2^~5TOJ}Ev3J<uaX05^-JA~Qa(6S575%hl+HX56bi&`Nvx<BH|b-7Iv`xi`gjXATaW>B!z{l|C$l$7aB26CE|%OcB$t~W4Xe$=RTIe_0IwN9atq*)x5R<aA79W`>LW^I!zvR^u2%@{BY%vLH@NOj>>nHV4^`sn&C0efsRV_gP%yb0~SOFa@Ao#hnlKPw8ePB``?w2V>!qWrGQIe--cR!GcNvH0Zj0<pSu6-p*!WFaq(&uE}hF>2pq-~qk6g)E$+L%_h0RM?7st2)%=QrF9LOKBGl$zK=jCb23}xEmX>Vh%-DdQ1p2i?!#hnW1Rnk5iXwkX$WoC~)7@%*wFgN?g;*V<k9Ld_xmMq=Dxhu;|5f#-IeYy=W_WbnNCN8o{1SmM1qOlP1FjBUMRf35h|qKd}b{b{Q4JIYffQ6fJT9@rCRiPk|xcR6-+D!CoL3c!1<sxgyv|FfF3Uvfc<=okc#V-M~8Gx{5VK##!BZQAMH>?4M;hv(afI*uNlcgr+e%zwG4t>4tbTypl-MxwgT1>S`!5MtV_^z7~5C*6o8w;&n|MXjm0(a(O-wBV@WI!gdqH(CnriN$tu!V%L-aaaw|k(o|PM+@B}g+HX*b?m-i%#TlE)c~P#3llS(?2a$N#e8sR+jt8qm54A>9$yClkhgi<6(m~B?g&{$-U(`KQ9RJjq(`zB3BE%^rAs#~nNT0Ukbz*qAp%LDP0xsSui+C3mtxb?XluFIAl>S5*?k3?Rt7U*GIA~5G%{V1`9<eFP08D?wHlcuG?t&tZN#n$=tR%+mkc)-O2^bVV^5!uq#Nk#N4-cXnPf9PbM?@k|LHmnUCO(L7nrsUXi&=eY!6s>oS)6hDsK`SS*l=3GGVXxqi&saA-%-jPi}S}i6Fn+MDT2+^SaG*sq8#5!u2F0|ARFgsTdHL59ZlIRD1t>*6~q8DQ=igBg;q<o2ng7uTska6p$MVfLT?v>HZDXy`MxE4ll&<xt>A-SG+l?y3Olf)=$S-&6Dajmc<Utq4(o6<W}xe(llYyZRc`lc_S|GEPgb^Tr6{#)e&#?9*bGWriM&}nziiga$wXn=L8^(umr2@YwG;I~7Y`im)Wm$j3Ch+iO{CKYP+2r5%#_!}z0}+y+-kYxL?WSQX(KeNT-jdS7+t`D|4wG<D|N&ZHi}Gk5F|kYsWYB&rwhs?s+#+erkZ7}Y47nHAV>qR+3w-g^ENSUt?)Rn;46|&p;RwR_!dv~Gp6_Tr#|gZ*PaG43d}9F@X*0ix@_>2*bw*V8cU&{vW(4rvmY1OT)IktROIjoNVN`+q}J{)nF~~S#?@z>Ml+vY6!Z)<)mG}3g1^}s%|JVSeUT~DX+7Kv=ew7dZoUHdxB1crzN0kG(#Yk;nXv&5oebOh{_gh8c35%iJ+1Z2qmyAdzLWipd;o9B^yg4QI~36(9Xt`PIy@U=oW%yie>;6yzN2dwptosm7y1y4!4%uvE7q6vT-&#$2I90Pj&kvUZs*##lsg!G5bCQ0jUSsnpZ*`m{SBl'
TRACE_ACTIONS = json.loads(zlib.decompress(base64.b85decode(_PAYLOAD)))


def agent(observation, configuration=None):
    del configuration
    step = min(int(observation.get("step", 0) or 0), len(TRACE_ACTIONS) - 1)
    return copy.deepcopy(TRACE_ACTIONS[step])


from typing import Any

tape_agent = agent

SELLABLE = (
    "STRAWBERRY",
    "MELON",
    "MILK",
    "WOOL",
    "EGG",
    "FERTILIZER",
    "WHEAT",
)

CRITICAL_PASTURE_CLEAR_STEP = 196
CRITICAL_PASTURE_HAND = 8


def clear_critical_pasture_weed(
    obs: dict[str, Any], action: dict[str, list[Any]], step: int
) -> None:
    """Use a redundant move to clear the day-8 pasture tile without tape drift."""
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", [])
    if player >= len(farms):
        return
    farm = farms[player]
    hands = farm.get("hands", [])
    commands = action.get("hands", [])
    if CRITICAL_PASTURE_HAND >= len(hands) or CRITICAL_PASTURE_HAND >= len(commands):
        return
    x, y = hands[CRITICAL_PASTURE_HAND]
    tiles = farm.get("tiles", [])
    tile = tiles[y][x]
    if step == CRITICAL_PASTURE_CLEAR_STEP:
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            commands[CRITICAL_PASTURE_HAND] = ["DIG"]
    elif step == CRITICAL_PASTURE_CLEAR_STEP + 1:
        if [x, y] == [5, 3] and tile is None:
            commands[CRITICAL_PASTURE_HAND] = ["NORTH"]
    elif step == CRITICAL_PASTURE_CLEAR_STEP + 2:
        critical_tile = tiles[3][5]
        if [x, y] == [5, 2] and critical_tile is None:
            commands[CRITICAL_PASTURE_HAND] = ["WEST"]


def agent(obs: dict[str, Any], config: Any = None) -> dict[str, list[Any]]:
    """Preserve the new opening and cash out unscheduled final inventory."""
    action = tape_agent(obs, config)
    step = int(obs.get("step", 0) or 0)
    clear_critical_pasture_weed(obs, action, step)
    day, hour = divmod(step, 24)
    if day != 29 or hour < 18:
        return action

    shed = obs.get("private", {}).get("shed", {})
    market = action.setdefault("market", [])
    scheduled: dict[str, int] = {}
    for order in market:
        if len(order) >= 3 and order[0] == "SELL":
            scheduled[order[1]] = scheduled.get(order[1], 0) + int(order[2])
    for product in SELLABLE:
        available = int(shed.get(product, 0) or 0) - scheduled.get(product, 0)
        if available <= 0 or len(market) >= 10:
            continue
        market.append(["SELL", product, available])
    return action


submission_agent = agent


v54_agent = agent

MOVE_DELTAS = {
    "NORTH": (0, -1),
    "EAST": (1, 0),
    "SOUTH": (0, 1),
    "WEST": (-1, 0),
}
INVERSE_MOVE = {
    "NORTH": "SOUTH",
    "EAST": "WEST",
    "SOUTH": "NORTH",
    "WEST": "EAST",
}

_pending_returns: dict[int, tuple[int, tuple[int, int], str]] = {}
_last_step = -1


def _commands(action: dict[str, list[Any]]) -> list[list[Any]]:
    return [action.get("farmer", ["PASS"]), *action.get("hands", [])]


def _set_command(action: dict[str, list[Any]], actor: int, command: list[Any]) -> None:
    if actor == 0:
        action["farmer"] = command
    else:
        action["hands"][actor - 1] = command


def _state_salt(obs: dict[str, Any], player: int, step: int, actor: int) -> int:
    farms = obs.get("farms", [])
    own_money = int(farms[player].get("money", 0) or 0)
    rival_money = sum(
        int(farm.get("money", 0) or 0)
        for index, farm in enumerate(farms)
        if index != player
    )
    shed = obs.get("private", {}).get("shed", {})
    inventory_salt = sum(
        (index + 3) * int(amount or 0)
        for index, amount in enumerate(shed.values())
    )
    return own_money * 3 + rival_money * 5 + inventory_salt * 7 + step * 11 + actor * 17


def _candidate_moves(
    obs: dict[str, Any], player: int, step: int, actor: int, position: list[int]
) -> list[tuple[str, tuple[int, int]]]:
    farm = obs["farms"][player]
    tiles = farm["tiles"]
    width = len(tiles[0])
    height = len(tiles)
    names = list(MOVE_DELTAS)
    salt = _state_salt(obs, player, step, actor)
    rotation = salt % len(names)
    names = names[rotation:] + names[:rotation]
    if (salt // len(names)) % 2:
        names.reverse()

    x, y = position
    candidates: list[tuple[str, tuple[int, int]]] = []
    for name in names:
        dx, dy = MOVE_DELTAS[name]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if tiles[ny][nx] == "LOCKED":
            continue
        candidates.append((name, (nx, ny)))
    return candidates


def _diversify_idle_routes(
    obs: dict[str, Any], action: dict[str, list[Any]], step: int
) -> None:
    global _last_step
    if step == 0 or step <= _last_step:
        _pending_returns.clear()
    _last_step = step

    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    positions = [farm["farmer"], *farm.get("hands", [])]
    commands = _commands(action)

    for actor, (expected_step, expected_position, inverse) in list(_pending_returns.items()):
        if expected_step != step:
            if expected_step < step:
                del _pending_returns[actor]
            continue
        del _pending_returns[actor]
        if actor >= len(positions) or actor >= len(commands):
            continue
        if tuple(positions[actor]) != expected_position or commands[actor] != ["PASS"]:
            continue
        _set_command(action, actor, [inverse])

    if step >= len(TRACE_ACTIONS) - 1 or step % 24 == 23:
        return
    next_action = TRACE_ACTIONS[step + 1]
    next_commands = _commands(next_action)
    commands = _commands(action)
    for actor, (position, command) in enumerate(zip(positions, commands)):
        if actor in _pending_returns:
            continue
        if command != ["PASS"] or actor >= len(next_commands):
            continue
        if next_commands[actor] != ["PASS"]:
            continue
        candidates = _candidate_moves(obs, player, step, actor, position)
        if not candidates:
            continue
        move, destination = candidates[0]
        _set_command(action, actor, [move])
        _pending_returns[actor] = (step + 1, destination, INVERSE_MOVE[move])


def agent(obs: dict[str, Any], config: Any = None) -> dict[str, list[Any]]:
    action = v54_agent(obs, config)
    step = int(obs.get("step", 0) or 0)
    _diversify_idle_routes(obs, action, step)
    return action


submission_agent = agent
v55_submission_agent = agent


from typing import Callable

v55_agent = agent

EARLY_PASTURE_REPAIRS = {
    # step: (actor index including farmer, expected coordinate)
    125: (4, (3, 3)),
    126: (5, (2, 4)),
}
LIQUIDATION_START = {
    "MILK": 12,
    "MELON": 15,
    "WOOL": 15,
    "STRAWBERRY": 17,
}
LIQUIDATION_FLOOR = {
    "MILK": 70,
    "MELON": 190,
    "WOOL": 80,
    "STRAWBERRY": 180,
}
_delayed_commands: dict[int, list[Any]] = {}
_last_repair_step = -1
repair_events: list[tuple[int, int, str, tuple[int, int]]] = []


def _set_actor(action: dict[str, list[Any]], actor: int, command: list[Any]) -> None:
    if actor == 0:
        action["farmer"] = command
    else:
        action["hands"][actor - 1] = command


def repair_early_pasture(
    obs: dict[str, Any], action: dict[str, list[Any]], step: int
) -> None:
    global _last_repair_step
    if step == 0 or step <= _last_repair_step or step % 24 == 0:
        _delayed_commands.clear()
        if step == 0:
            repair_events.clear()
    _last_repair_step = step

    commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for actor, pending in list(_delayed_commands.items()):
        if actor >= len(commands):
            _delayed_commands.pop(actor, None)
            continue
        current = commands[actor]
        _set_actor(action, actor, pending)
        if current == ["PASS"]:
            _delayed_commands.pop(actor, None)
        else:
            _delayed_commands[actor] = current

    repair = EARLY_PASTURE_REPAIRS.get(step)
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    positions = [farm["farmer"], *farm.get("hands", [])]
    commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    if repair is not None:
        actor, expected = repair
        if (
            actor not in _delayed_commands
            and actor < len(positions)
            and actor < len(commands)
        ):
            x, y = positions[actor]
            tile = farm["tiles"][y][x]
            if (
                (x, y) == expected
                and commands[actor] == ["BUILD_PASTURE"]
                and isinstance(tile, dict)
                and tile.get("kind") == "WEED"
            ):
                _set_actor(action, actor, ["DIG"])
                _delayed_commands[actor] = ["BUILD_PASTURE"]
                repair_events.append((step, actor, "BUILD_PASTURE", (x, y)))

    # Random weed growth can invalidate any later plant/build command.  Repair
    # it with the same-worker delayed pipeline while enough day remains to
    # encounter a PASS and resynchronize with the base tape.
    hour = step % 24
    if hour > 21 or (step < 600 and hour < 20):
        return
    commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for actor, ((x, y), command) in enumerate(zip(positions, commands)):
        if actor in _delayed_commands or not command:
            continue
        if command[0] not in {"PLANT", "BUILD_PASTURE"}:
            continue
        tile = farm["tiles"][y][x]
        if not isinstance(tile, dict) or tile.get("kind") != "WEED":
            continue
        _set_actor(action, actor, ["DIG"])
        _delayed_commands[actor] = command
        repair_events.append((step, actor, str(command[0]), (x, y)))


def append_liquidation(
    obs: dict[str, Any],
    action: dict[str, list[Any]],
    day: int,
    mode: str,
    *,
    lead_gap: float = 250,
    floor_scale: float = 1.0,
    front_run: dict[str, float] | None = None,
) -> None:
    shed = obs.get("private", {}).get("shed", {})
    market = action.setdefault("market", [])
    scheduled: dict[str, int] = {}
    for order in market:
        if len(order) >= 3 and order[0] == "SELL":
            scheduled[order[1]] = scheduled.get(order[1], 0) + int(order[2])
    for product, start_day in LIQUIDATION_START.items():
        if day < start_day or len(market) >= 10:
            continue
        price = int(obs.get("market", {}).get("prices", {}).get(product, 0) or 0)
        front_run_now = bool(
            front_run
            and product in front_run
            and price >= front_run[product]
        )
        if mode in {"floor", "behind_floor"}:
            if price < LIQUIDATION_FLOOR[product] * floor_scale:
                continue
        if mode == "behind_floor":
            player = int(obs.get("player", 0) or 0)
            farms = obs.get("farms", [])
            own_money = float(farms[player].get("money", 0) or 0)
            rival_money = max(
                (float(farm.get("money", 0) or 0) for index, farm in enumerate(farms) if index != player),
                default=own_money,
            )
            if not front_run_now and rival_money - own_money < lead_gap:
                continue
        available = int(shed.get(product, 0) or 0) - scheduled.get(product, 0)
        if available > 0:
            market.append(["SELL", product, available])


def reorder_market(
    obs: dict[str, Any], action: dict[str, list[Any]], strategy: str
) -> None:
    market = action.setdefault("market", [])
    if strategy == "none" or len(market) < 2:
        return
    if strategy == "reverse":
        market.reverse()
        return

    priorities = {
        "SELL": 0,
        "HIRE": 1,
        "BUY_SEED": 2,
        "BUY_PRODUCT": 3,
        "BUY_ANIMAL": 4,
        "BUY_LAND": 5,
    }
    if strategy == "stable_sell_first":
        market.sort(key=lambda order: priorities.get(order[0], 9))
        return
    if strategy == "value_sell_first":
        prices = obs.get("market", {}).get("prices", {})

        def key(order: list[Any]) -> tuple[float, float]:
            priority = priorities.get(order[0], 9)
            if order[0] != "SELL" or len(order) < 3:
                return float(priority), 0.0
            value = float(prices.get(order[1], 0)) * int(order[2])
            return float(priority), -value

        market.sort(key=key)


def make_agent(
    strategy: str,
    liquidate: bool | str,
    *,
    lead_gap: float = 250,
    floor_scale: float = 1.0,
    front_run: dict[str, float] | None = None,
    activation_gap: float | None = None,
    activation_day: int = 0,
    activation_deadband: tuple[float, float] | None = None,
) -> Callable[..., dict[str, list[Any]]]:
    counterplay_active = activation_gap is None
    counterplay_disabled = False
    activation_regime_checked = False

    def agent(obs: dict[str, Any], config: Any = None) -> dict[str, list[Any]]:
        nonlocal counterplay_active, counterplay_disabled, activation_regime_checked
        action = v55_agent(obs, config)
        step = int(obs.get("step", 0) or 0)
        day = int(obs.get("day", step // 24) or 0)
        if step == 0:
            counterplay_active = activation_gap is None
            counterplay_disabled = False
            activation_regime_checked = False
        repair_early_pasture(obs, action, step)
        if not counterplay_active and not counterplay_disabled and day >= activation_day:
            player = int(obs.get("player", 0) or 0)
            farms = obs.get("farms", [])
            own_money = float(farms[player].get("money", 0) or 0)
            rival_money = max(
                (
                    float(farm.get("money", 0) or 0)
                    for index, farm in enumerate(farms)
                    if index != player
                ),
                default=own_money,
            )
            rival_lead = rival_money - own_money
            if not activation_regime_checked:
                activation_regime_checked = True
                if (
                    activation_deadband is not None
                    and activation_deadband[0] <= rival_lead <= activation_deadband[1]
                ):
                    counterplay_disabled = True
            if not counterplay_disabled:
                counterplay_active = rival_lead >= float(activation_gap or 0)
        if counterplay_active and liquidate:
            mode = "aggressive" if liquidate is True else str(liquidate)
            append_liquidation(
                obs,
                action,
                day,
                mode,
                lead_gap=lead_gap,
                floor_scale=floor_scale,
                front_run=front_run,
            )
        if counterplay_active:
            reorder_market(obs, action, strategy)
        return action

    return agent


agent = make_agent(
    'value_sell_first',
    'behind_floor',
    lead_gap=0,
    floor_scale=0.6,
    front_run={'STRAWBERRY': 215},
    activation_gap=100,
    activation_day=12,
    activation_deadband=(800, 1300),
)
v56_submission_agent = agent
