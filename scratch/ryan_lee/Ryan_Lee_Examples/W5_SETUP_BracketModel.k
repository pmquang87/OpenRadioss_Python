$# LS-DYNA Keyword file created by LS-PrePost(R) V4.5.15 - 08Mar2018
$# Created on Apr-07-2018 (21:52:44)
*KEYWORD
*TITLE
$#                                                                         title
LS-DYNA keyword deck by LS-PrePost
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol     
       1.0         0       0.0       0.01.000000E8         0
*DATABASE_GLSTAT
$#      dt    binary      lcur     ioopt     
      0.02         0         0         1
*DATABASE_MATSUM
$#      dt    binary      lcur     ioopt     
      0.02         0         0         1
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid      
       0.1         0         0         0         0
$#   ioopt     
         0
*BOUNDARY_SPC_SET
$#    nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz
         2         0         1         1         1         1         1         1
*SET_NODE_LIST_TITLE
NODESET(SPC) 2
$#     sid       da1       da2       da3       da4    solver      
         2       0.0       0.0       0.0       0.0MECH
$#    nid1      nid2      nid3      nid4      nid5      nid6      nid7      nid8
      1688      1690      1691      1692      1693      1695      1696      1697
      1689      1694      1874      1875      1876      1877      1878      1879
      1880      1881      1882      1883       540       541       542       543
       544       545       546       547       548       549      2062      2063
      2064      2065      2066      2067      2068      2069      2070      2071
*LOAD_SEGMENT_ID
$#      id                                                               heading
         1                                                                      
$#    lcid        sf        at        n1        n2        n3        n4        n5
         11.000000E7       0.0      1118       684       834       676         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         2                                                                      
         11.000000E7       0.0       684      1119       890       834         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         3                                                                      
         11.000000E7       0.0      1119       682       785       890         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         4                                                                      
         11.000000E7       0.0       785      1132       682       682         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         5                                                                      
         11.000000E7       0.0      1132      1134       787       785         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         6                                                                      
         11.000000E7       0.0      1134       687       848       787         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         7                                                                      
         11.000000E7       0.0       687      1131      1154       848         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         8                                                                      
         11.000000E7       0.0      1211       707       836       699         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
         9                                                                      
         11.000000E7       0.0       707       706       910       836         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        10                                                                      
         11.000000E7       0.0       706       705       791       910         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        11                                                                      
         11.000000E7       0.0       791       708       705       705         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        12                                                                      
         11.000000E7       0.0       708      1227       793       791         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        13                                                                      
         11.000000E7       0.0      1227      1226       820       793         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        14                                                                      
         11.000000E7       0.0      1226       994      1002       820         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        15                                                                      
         11.000000E7       0.0       937       939       808       444         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        16                                                                      
         11.000000E7       0.0       939       637       796       808         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        17                                                                      
         11.000000E7       0.0       637       936       797       796         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        18                                                                      
         11.000000E7       0.0       936       639       797       797         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        19                                                                      
         11.000000E7       0.0       639       642       799       797         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        20                                                                      
         11.000000E7       0.0       642       613       861       799         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        21                                                                      
         11.000000E7       0.0       613       611       367       861         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        22                                                                      
         11.000000E7       0.0      1042      1039       672       821         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        23                                                                      
         11.000000E7       0.0      1043      1042       821       891         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        24                                                                      
         11.000000E7       0.0      1041      1043       891       804         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        25                                                                      
         11.000000E7       0.0       804      1041      1024      1024         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        26                                                                      
         11.000000E7       0.0       660      1024       804       805         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        27                                                                      
         11.000000E7       0.0      1028       660       805       857         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        28                                                                      
         11.000000E7       0.0       952      1028       857       974         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        29                                                                      
         11.000000E7       0.0       443       444       808       807         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        30                                                                      
         11.000000E7       0.0       442       443       807       809         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        31                                                                      
         11.000000E7       0.0       441       442       809       810         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        32                                                                      
         11.000000E7       0.0       440       441       810       811         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        33                                                                      
         11.000000E7       0.0       630       440       811       812         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        34                                                                      
         11.000000E7       0.0       627       630       812       813         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        35                                                                      
         11.000000E7       0.0       813       862       997       996         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        36                                                                      
         11.000000E7       0.0       998       997       862       816         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        37                                                                      
         11.000000E7       0.0       999       998       816       817         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        38                                                                      
         11.000000E7       0.0       624       999       817       863         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        39                                                                      
         11.000000E7       0.0      1001       624       863       819         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        40                                                                      
         11.000000E7       0.0      1002      1001       819       820         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        41                                                                      
         11.000000E7       0.0       672       671       822       821         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        42                                                                      
         11.000000E7       0.0       671      1059       872       822         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        43                                                                      
         11.000000E7       0.0      1059       669       824       872         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        44                                                                      
         11.000000E7       0.0       669       668       825       824         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        45                                                                      
         11.000000E7       0.0       668       667       826       825         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        46                                                                      
         11.000000E7       0.0       667      1055       827       826         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        47                                                                      
         11.000000E7       0.0       829       827      1055       674         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        48                                                                      
         11.000000E7       0.0       681       680       830       829         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        49                                                                      
         11.000000E7       0.0       680       679       831       830         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        50                                                                      
         11.000000E7       0.0       679       678       832       831         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        51                                                                      
         11.000000E7       0.0       678       677       833       832         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        52                                                                      
         11.000000E7       0.0       677       676       834       833         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        53                                                                      
         11.000000E7       0.0       700       699       836       835         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        54                                                                      
         11.000000E7       0.0       701       700       835       837         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        55                                                                      
         11.000000E7       0.0       702       701       837       838         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        56                                                                      
         11.000000E7       0.0       703       702       838       839         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        57                                                                      
         11.000000E7       0.0      1202       703       839       840         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        58                                                                      
         11.000000E7       0.0       696      1202       840       841         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        59                                                                      
         11.000000E7       0.0       841       843      1149       689         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        60                                                                      
         11.000000E7       0.0      1150      1149       843       844         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        61                                                                      
         11.000000E7       0.0      1151      1150       844       845         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        62                                                                      
         11.000000E7       0.0       693      1151       845       846         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        63                                                                      
         11.000000E7       0.0      1153       693       846       847         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        64                                                                      
         11.000000E7       0.0      1154      1153       847       848         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        65                                                                      
         11.000000E7       0.0       849       366       651       650         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        66                                                                      
         11.000000E7       0.0       650       979       850       849         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        67                                                                      
         11.000000E7       0.0       372       366       849       851         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        68                                                                      
         11.000000E7       0.0       979       978       852       850         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        69                                                                      
         11.000000E7       0.0       648       372       851       853         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        70                                                                      
         11.000000E7       0.0       978       977       854       852         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        71                                                                      
         11.000000E7       0.0       977       655       855       854         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        72                                                                      
         11.000000E7       0.0       655       975       856       855         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        73                                                                      
         11.000000E7       0.0       975       974       857       856         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        74                                                                      
         11.000000E7       0.0       370       648       853       858         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        75                                                                      
         11.000000E7       0.0       369       370       858       859         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        76                                                                      
         11.000000E7       0.0       645       369       859       860         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        77                                                                      
         11.000000E7       0.0       367       645       860       861         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        78                                                                      
         11.000000E7       0.0       768       771       850       852         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        79                                                                      
         11.000000E7       0.0       849       850       771       851         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        80                                                                      
         11.000000E7       0.0       851       771       770       853         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        81                                                                      
         11.000000E7       0.0       853       770       769       858         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        82                                                                      
         11.000000E7       0.0       760       763       840       839         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        83                                                                      
         11.000000E7       0.0       753       751       816       862         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        84                                                                      
         11.000000E7       0.0       754       753       812       811         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        85                                                                      
         11.000000E7       0.0       753       862       813       812         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        86                                                                      
         11.000000E7       0.0       813       996       628       627         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        87                                                                      
         11.000000E7       0.0       754       811       810       755         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        88                                                                      
         11.000000E7       0.0       758       757       864       863         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        89                                                                      
         11.000000E7       0.0       751       758       863       817         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        90                                                                      
         11.000000E7       0.0       816       751       817       817         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        91                                                                      
         11.000000E7       0.0       765       764       866       865         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        92                                                                      
         11.000000E7       0.0       766       765       865       867         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        93                                                                      
         11.000000E7       0.0       775       824       825       825         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        94                                                                      
         11.000000E7       0.0       777       826       827       829         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        95                                                                      
         11.000000E7       0.0       778       777       829       830         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        96                                                                      
         11.000000E7       0.0       829       681       673       674         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        97                                                                      
         11.000000E7       0.0       778       830       831       779         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        98                                                                      
         11.000000E7       0.0       773       772       870       869         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
        99                                                                      
         11.000000E7       0.0       774       773       869       871         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       100                                                                      
         11.000000E7       0.0       762       761       845       844         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       101                                                                      
         11.000000E7       0.0       763       762       844       843         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       102                                                                      
         11.000000E7       0.0       840       763       843       841         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       103                                                                      
         11.000000E7       0.0       841       689       697       696         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       104                                                                      
         11.000000E7       0.0       782       781       873       872         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       105                                                                      
         11.000000E7       0.0       775       782       872       824         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       106                                                                      
         11.000000E7       0.0       780       776       875       874         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       107                                                                      
         11.000000E7       0.0       781       780       874       876         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       108                                                                      
         11.000000E7       0.0       756       752       878       877         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       109                                                                      
         11.000000E7       0.0       757       756       877       879         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       110                                                                      
         11.000000E7       0.0       759       766       881       880         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       111                                                                      
         11.000000E7       0.0       761       759       880       882         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       112                                                                      
         11.000000E7       0.0       845       761       882       846         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       113                                                                      
         11.000000E7       0.0       883       752       755       810         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       114                                                                      
         11.000000E7       0.0       837       866       764       838         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       115                                                                      
         11.000000E7       0.0       760       839       838       764         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       116                                                                      
         11.000000E7       0.0       769       767       885       884         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       117                                                                      
         11.000000E7       0.0       737       736       874       875         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       118                                                                      
         11.000000E7       0.0       736       735       876       874         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       119                                                                      
         11.000000E7       0.0       738       737       875       886         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       120                                                                      
         11.000000E7       0.0       739       738       886       887         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       121                                                                      
         11.000000E7       0.0       735       734       923       876         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       122                                                                      
         11.000000E7       0.0       734       733       889       923         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       123                                                                      
         11.000000E7       0.0       740       739       887       890         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       124                                                                      
         11.000000E7       0.0       741       740       890       785         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       125                                                                      
         11.000000E7       0.0       733       732       891       889         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       126                                                                      
         11.000000E7       0.0       732       712       804       891         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       127                                                                      
         11.000000E7       0.0       781       876       923       873         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       128                                                                      
         11.000000E7       0.0       805       804       712       731         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       129                                                                      
         11.000000E7       0.0       787       742       741       785         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       130                                                                      
         11.000000E7       0.0       767       774       894       885         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       131                                                                      
         11.000000E7       0.0       805       731       730       896         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       132                                                                      
         11.000000E7       0.0       730       729       897       896         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       133                                                                      
         11.000000E7       0.0       729       728       869       897         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       134                                                                      
         11.000000E7       0.0       869       728       727       871         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       135                                                                      
         11.000000E7       0.0       871       727       894       774         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       136                                                                      
         11.000000E7       0.0       727       726       898       894         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       137                                                                      
         11.000000E7       0.0       726       725       899       898         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       138                                                                      
         11.000000E7       0.0       725       724       900       899         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       139                                                                      
         11.000000E7       0.0       724       723       799       900         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       140                                                                      
         11.000000E7       0.0       723       722       797       799         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       141                                                                      
         11.000000E7       0.0       796       797       722       721         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       142                                                                      
         11.000000E7       0.0       885       894       898       899         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       143                                                                      
         11.000000E7       0.0       787       903       743       742         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       144                                                                      
         11.000000E7       0.0       744       743       903       905         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       145                                                                      
         11.000000E7       0.0       745       744       905       906         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       146                                                                      
         11.000000E7       0.0       746       745       906       881         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       147                                                                      
         11.000000E7       0.0       746       867       747       747         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       148                                                                      
         11.000000E7       0.0       881       766       867       746         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       149                                                                      
         11.000000E7       0.0       747       867       865       907         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       150                                                                      
         11.000000E7       0.0       748       747       907       907         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       151                                                                      
         11.000000E7       0.0       749       748       907       908         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       152                                                                      
         11.000000E7       0.0       750       749       908       910         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       153                                                                      
         11.000000E7       0.0       711       750       910       791         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       154                                                                      
         11.000000E7       0.0       910       908       927       836         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       155                                                                      
         11.000000E7       0.0       793       713       711       791         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       156                                                                      
         11.000000E7       0.0       796       721       720       913         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       157                                                                      
         11.000000E7       0.0       720       719       914       913         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       158                                                                      
         11.000000E7       0.0       719       718       878       914         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       159                                                                      
         11.000000E7       0.0       718       717       877       878         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       160                                                                      
         11.000000E7       0.0       717       716       879       877         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       161                                                                      
         11.000000E7       0.0       716       715       915       879         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       162                                                                      
         11.000000E7       0.0       715       714       916       915         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       163                                                                      
         11.000000E7       0.0       793       916       714       713         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       164                                                                      
         11.000000E7       0.0       757       879       915       864         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       165                                                                      
         11.000000E7       0.0       881       906       905       880         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       166                                                                      
         11.000000E7       0.0       854       855       870       772         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       167                                                                      
         11.000000E7       0.0       854       772       768       852         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       168                                                                      
         11.000000E7       0.0       855       856       919       870         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       169                                                                      
         11.000000E7       0.0       869       870       919       897         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       170                                                                      
         11.000000E7       0.0       819       863       864       921         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       171                                                                      
         11.000000E7       0.0       769       884       859       858         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       172                                                                      
         11.000000E7       0.0       860       859       884       922         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       173                                                                      
         11.000000E7       0.0       884       885       899       922         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       174                                                                      
         11.000000E7       0.0       752       883       914       878         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       175                                                                      
         11.000000E7       0.0       873       923       821       822         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       176                                                                      
         11.000000E7       0.0       891       821       923       889         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       177                                                                      
         11.000000E7       0.0       820       819       921       924         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       178                                                                      
         11.000000E7       0.0       793       820       924       916         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       179                                                                      
         11.000000E7       0.0       883       807       809       810         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       180                                                                      
         11.000000E7       0.0       921       864       915       924         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       181                                                                      
         11.000000E7       0.0       776       779       925       925         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       182                                                                      
         11.000000E7       0.0       875       776       925       886         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       183                                                                      
         11.000000E7       0.0       865       866       926       907         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       184                                                                      
         11.000000E7       0.0       908       907       926       927         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       185                                                                      
         11.000000E7       0.0       861       860       922       928         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       186                                                                      
         11.000000E7       0.0       799       861       928       900         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       187                                                                      
         11.000000E7       0.0       856       857       929       919         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       188                                                                      
         11.000000E7       0.0       857       805       896       929         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       189                                                                      
         11.000000E7       0.0       808       796       913       930         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       190                                                                      
         11.000000E7       0.0       913       914       883       930         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       191                                                                      
         11.000000E7       0.0       787       848       932       903         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       192                                                                      
         11.000000E7       0.0       905       903       932       880         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       193                                                                      
         11.000000E7       0.0       848       847       933       932         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       194                                                                      
         11.000000E7       0.0       847       846       882       933         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       195                                                                      
         11.000000E7       0.0       807       808       930       883         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       196                                                                      
         11.000000E7       0.0       880       932       933       882         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       197                                                                      
         11.000000E7       0.0       928       922       899       900         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       198                                                                      
         11.000000E7       0.0       919       929       896       897         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       199                                                                      
         11.000000E7       0.0       822       872       873       873         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       200                                                                      
         11.000000E7       0.0       916       924       915       915         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       201                                                                      
         11.000000E7       0.0       835       926       866       837         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       202                                                                      
         11.000000E7       0.0       835       836       927       926         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       203                                                                      
         11.000000E7       0.0       779       831       832       925         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       204                                                                      
         11.000000E7       0.0       832       833       934       925         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       205                                                                      
         11.000000E7       0.0       833       834       935       934         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       206                                                                      
         11.000000E7       0.0       834       890       887       935         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       207                                                                      
         11.000000E7       0.0       935       887       886       886         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       208                                                                      
         11.000000E7       0.0       886       925       934       935         0
*LOAD_SEGMENT_ID
$#      id                                                               heading
       209                                                                      
         11.000000E7       0.0       825       826       777       775         0
*PART
$#                                                                         title
LSHELL1
$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         1         1         1         0         0         0         0         0
*SECTION_SHELL
$#   secid    elform      shrf       nip     propt   qr/irid     icomp     setyp
         1         2       1.0         2       1.0         0         0         1
$#      t1        t2        t3        t4      nloc     marea      idof    edgset
     0.002     0.002     0.002     0.002       0.0       0.0       0.0         0
*MAT_ELASTIC
$#     mid        ro         e        pr        da        db  not used        
         1    2580.07.20000E10       0.3       0.0       0.0         0
*DEFINE_CURVE
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint
         1         0       1.0       1.0       0.0       0.0         0         0
$#                a1                  o1  
                 0.0                 0.0
                 1.0                 1.0
                 1.1                 1.0
*SET_NODE_LIST_TITLE
NODESET(SPC) 1
$#     sid       da1       da2       da3       da4    solver      
         1       0.0       0.0       0.0       0.0MECH
$#    nid1      nid2      nid3      nid4      nid5      nid6      nid7      nid8
      1688      1689      1690      1691      1692      1693      1694      1695
      1696      1697         0         0         0         0         0         0
*ELEMENT_SHELL
$#   eid     pid      n1      n2      n3      n4      n5      n6      n7      n8
       1       1      20       4      40       1       0       0       0       0
       2       1      11      13      50       3       0       0       0       0
       3       1      19      42       4      20       0       0       0       0
       4       1      10      72      13      11       0       0       0       0
       5       1      76      43      42      19       0       0       0       0
       6       1      46      15      72      10       0       0       0       0
       7       1      17       7      43      76       0       0       0       0
       8       1      45      74      15      46       0       0       0       0
       9       1      74      45       7      17       0       0       0       0
      10       1      28     103       1      40       0       0       0       0
      11       1      24      27      25     101       0       0       0       0
      12       1      27      24     103      28       0       0       0       0
      13       1       4      31     126      40       0       0       0       0
      14       1      38      11       3     106       0       0       0       0
      15       1     135      10      11      38       0       0       0       0
      16       1      42     130      31       4       0       0       0       0
      17       1      43     131     130      42       0       0       0       0
      18       1      36      46      10     135       0       0       0       0
      19       1       7      34     131      43       0       0       0       0
      20       1     133      45      46      36       0       0       0       0
      21       1      34       7      45     133       0       0       0       0
      22       1     142     144     113     153       0       0       0       0
      23       1       3      50      53      58       0       0       0       0
      24       1     153     113     114      52       0       0       0       0
      25       1      58      53      52     114       0       0       0       0
      26       1      68      20       1     221       0       0       0       0
      27       1      13     169     146      50       0       0       0       0
      28       1      72      62     169      13       0       0       0       0
      29       1     175      19      20      68       0       0       0       0
      30       1     174      76      19     175       0       0       0       0
      31       1      15      63      62      72       0       0       0       0
      32       1     173      17      76     174       0       0       0       0
      33       1      74     172      63      15       0       0       0       0
      34       1      17     173     172      74       0       0       0       0
      35       1      27     198      80      25       0       0       0       0
      36       1      81      28      40     126       0       0       0       0
      37       1     198      27      28      81       0       0       0       0
      38       1      90      94     185     211       0       0       0       0
      39       1     188      91     101      25       0       0       0       0
      40       1      91     188      94      90       0       0       0       0
      41       1      99      24     101     411       0       0       0       0
      42       1     103      98     221       1       0       0       0       0
      43       1      98     103      24      99       0       0       0       0
      44       1     244     113     144     463       0       0       0       0
      45       1      58     109     106       3       0       0       0       0
      46       1     114     245     109      58       0       0       0       0
      47       1     245     114     113     244       0       0       0       0
      48       1     199     126      31     118       0       0       0       0
      49       1     199     118     116     116       0       0       0       0
      50       1     242      38     106     106       0       0       0       0
      51       1     242     117     273      38       0       0       0       0
      52       1     135      38     273     272       0       0       0       0
      53       1      36     135     272     271       0       0       0       0
      54       1     133      36     271     270       0       0       0       0
      55       1      34     133     270     121       0       0       0       0
      56       1     121     268     131      34       0       0       0       0
      57       1     130     131     268     267       0       0       0       0
      58       1     267     118      31     130       0       0       0       0
      59       1     235     288     283     458       0       0       0       0
      60       1     288     235     144     142       0       0       0       0
      61       1     153     305     495     142       0       0       0       0
      62       1     148      53      50     146       0       0       0       0
      63       1      52     304     305     153       0       0       0       0
      64       1     304      52      53     148       0       0       0       0
      65       1      68     221     222     165       0       0       0       0
      66       1     222     321     165     165       0       0       0       0
      67       1     156     306     169     323       0       0       0       0
      68       1     306     146     169     169       0       0       0       0
      69       1     324     323     169      62       0       0       0       0
      70       1     160     324      62      63       0       0       0       0
      71       1      63     172     161     160       0       0       0       0
      72       1     172     173     162     161       0       0       0       0
      73       1     328     162     173     174       0       0       0       0
      74       1     174     175     164     328       0       0       0       0
      75       1     165     164     175      68       0       0       0       0
      76       1      94     182     609     185       0       0       0       0
      77       1     182     188      25      80       0       0       0       0
      78       1      94     188     182     182       0       0       0       0
      79       1     357     346     200     191       0       0       0       0
      80       1     346      80     198     200       0       0       0       0
      81       1     126     199     200      81       0       0       0       0
      82       1     198      81     200     200       0       0       0       0
      83       1     199     116     192     200       0       0       0       0
      84       1     192     191     200     200       0       0       0       0
      85       1     398     343     340     201       0       0       0       0
      86       1     343     398     211     185       0       0       0       0
      87       1     209      90     211     207       0       0       0       0
      88       1      91     209     411     101       0       0       0       0
      89       1      90     209      91      91       0       0       0       0
      90       1     228     419     220     229       0       0       0       0
      91       1     411     228     229      99       0       0       0       0
      92       1     321     222     229     424       0       0       0       0
      93       1     220     424     229     229       0       0       0       0
      94       1     222     221      98     229       0       0       0       0
      95       1     229      98      99      99       0       0       0       0
      96       1     235     456     463     144       0       0       0       0
      97       1     456     235     458     458       0       0       0       0
      98       1     463     465     240     244       0       0       0       0
      99       1     465     237     240     240       0       0       0       0
     100       1     117     242     469     469       0       0       0       0
     101       1     242     106     109     469       0       0       0       0
     102       1     245     244     240     239       0       0       0       0
     103       1     109     245     239     469       0       0       0       0
     104       1     118     274     254     116       0       0       0       0
     105       1     274     256     253     254       0       0       0       0
     106       1     263     275     265     255       0       0       0       0
     107       1     275     273     117     265       0       0       0       0
     108       1     276     275     263     262       0       0       0       0
     109       1     277     274     118     267       0       0       0       0
     110       1     278     277     267     268       0       0       0       0
     111       1     279     278     268     121       0       0       0       0
     112       1     280     276     262     261       0       0       0       0
     113       1     281     279     121     270       0       0       0       0
     114       1     281     280     261     260       0       0       0       0
     115       1     279     281     260     259       0       0       0       0
     116       1     278     279     259     258       0       0       0       0
     117       1     277     278     258     257       0       0       0       0
     118       1     274     277     257     256       0       0       0       0
     119       1     280     281     270     271       0       0       0       0
     120       1     276     280     271     272       0       0       0       0
     121       1     275     276     272     273       0       0       0       0
     122       1     504     288     142     495       0       0       0       0
     123       1     504     283     288     288       0       0       0       0
     124       1     537     289     459     294       0       0       0       0
     125       1     458     283     294     459       0       0       0       0
     126       1     583     301     299     299       0       0       0       0
     127       1     301     495     305     299       0       0       0       0
     128       1     306     156     297     297       0       0       0       0
     129       1     146     306     297     148       0       0       0       0
     130       1     299     305     304     587       0       0       0       0
     131       1     304     148     297     587       0       0       0       0
     132       1     323     331     311     156       0       0       0       0
     133       1     331     313     589     311       0       0       0       0
     134       1     320     332     322     312       0       0       0       0
     135       1     332     165     321     322       0       0       0       0
     136       1     333     331     323     324       0       0       0       0
     137       1     334     332     320     319       0       0       0       0
     138       1     335     334     319     318       0       0       0       0
     139       1     336     335     318     317       0       0       0       0
     140       1     337     333     324     160       0       0       0       0
     141       1     338     336     317     316       0       0       0       0
     142       1     338     337     160     161       0       0       0       0
     143       1     336     338     161     162       0       0       0       0
     144       1     335     336     162     328       0       0       0       0
     145       1     334     335     328     164       0       0       0       0
     146       1     332     334     164     165       0       0       0       0
     147       1     337     338     316     315       0       0       0       0
     148       1     333     337     315     314       0       0       0       0
     149       1     331     333     314     313       0       0       0       0
     150       1     607     340     343     343       0       0       0       0
     151       1     607     343     185     609       0       0       0       0
     152       1     346     357     363     356       0       0       0       0
     153       1      80     346     356     182       0       0       0       0
     154       1     363     361     610     356       0       0       0       0
     155       1     610     609     182     356       0       0       0       0
     156       1     372     374     366     366       0       0       0       0
     157       1     374     372     373     373       0       0       0       0
     158       1     372     648     375     373       0       0       0       0
     159       1     380     375     648     370       0       0       0       0
     160       1     385     367     611     606       0       0       0       0
     161       1     386     385     606     361       0       0       0       0
     162       1     387     254     253     383       0       0       0       0
     163       1     387     192     116     254       0       0       0       0
     164       1     388     387     383     382       0       0       0       0
     165       1     389     388     382     376       0       0       0       0
     166       1     390     389     376     377       0       0       0       0
     167       1     386     390     377     378       0       0       0       0
     168       1     390     386     361     363       0       0       0       0
     169       1     389     390     363     357       0       0       0       0
     170       1     388     389     357     191       0       0       0       0
     171       1     387     388     191     192       0       0       0       0
     172       1     379     380     370     369       0       0       0       0
     173       1     391     386     378     379       0       0       0       0
     174       1     391     379     369     645       0       0       0       0
     175       1     385     391     645     367       0       0       0       0
     176       1     386     391     385     385       0       0       0       0
     177       1     392     398     201     201       0       0       0       0
     178       1     398     392     207     211       0       0       0       0
     179       1     201     340     404     402       0       0       0       0
     180       1     639     936     402     404       0       0       0       0
     181       1     207     407     415     209       0       0       0       0
     182       1     407     416     410     415       0       0       0       0
     183       1     411     209     415     228       0       0       0       0
     184       1     419     228     415     410       0       0       0       0
     185       1     630     627     438     438       0       0       0       0
     186       1     630     438     436     436       0       0       0       0
     187       1     630     436     431     440       0       0       0       0
     188       1     431     432     441     440       0       0       0       0
     189       1     444     445     418     937       0       0       0       0
     190       1     445     446     416     418       0       0       0       0
     191       1     322     447     430     312       0       0       0       0
     192       1     424     447     322     321       0       0       0       0
     193       1     447     448     429     430       0       0       0       0
     194       1     448     449     428     429       0       0       0       0
     195       1     449     450     435     428       0       0       0       0
     196       1     450     446     434     435       0       0       0       0
     197       1     446     450     410     416       0       0       0       0
     198       1     450     449     419     410       0       0       0       0
     199       1     449     448     220     419       0       0       0       0
     200       1     448     447     424     220       0       0       0       0
     201       1     432     433     442     441       0       0       0       0
     202       1     446     451     433     434       0       0       0       0
     203       1     433     451     443     442       0       0       0       0
     204       1     451     445     444     443       0       0       0       0
     205       1     446     445     451     451       0       0       0       0
     206       1     456     458     459     457       0       0       0       0
     207       1     459     289     536     457       0       0       0       0
     208       1     452     457     536     454       0       0       0       0
     209       1     457     465     463     456       0       0       0       0
     210       1     457     452     474     474       0       0       0       0
     211       1     457     474     237     465       0       0       0       0
     212       1     474     452     476     476       0       0       0       0
     213       1     237     474     476     485       0       0       0       0
     214       1     478     255     265     486       0       0       0       0
     215       1     265     117     469     486       0       0       0       0
     216       1     479     478     486     487       0       0       0       0
     217       1     480     479     487     488       0       0       0       0
     218       1     481     480     488     485       0       0       0       0
     219       1     482     481     485     485       0       0       0       0
     220       1     482     485     476     526       0       0       0       0
     221       1     485     488     240     237       0       0       0       0
     222       1     239     240     488     487       0       0       0       0
     223       1     469     239     487     486       0       0       0       0
     224       1     301     492     504     495       0       0       0       0
     225       1     507     492     494     494       0       0       0       0
     226       1     492     301     583     494       0       0       0       0
     227       1     283     504     492     294       0       0       0       0
     228       1     537     294     492     539       0       0       0       0
     229       1     492     507     500     539       0       0       0       0
     230       1     506     514     550     597       0       0       0       0
     231       1     507     597     550     500       0       0       0       0
     232       1     531     526     476     551       0       0       0       0
     233       1     476     452     454     551       0       0       0       0
     234       1     514     513     552     550       0       0       0       0
     235       1     513     512     578     552       0       0       0       0
     236       1     530     531     551     554       0       0       0       0
     237       1     529     530     554     566       0       0       0       0
     238       1     512     511     556     578       0       0       0       0
     239       1     528     529     566     557       0       0       0       0
     240       1     511     510     558     556       0       0       0       0
     241       1     527     528     557     559       0       0       0       0
     242       1     510     509     560     558       0       0       0       0
     243       1     509     517     516     560       0       0       0       0
     244       1     523     527     559     561       0       0       0       0
     245       1     524     523     561     525       0       0       0       0
     246       1     500     550     552     562       0       0       0       0
     247       1     539     500     562     563       0       0       0       0
     248       1     537     539     563     546       0       0       0       0
     249       1     537     546     541     289       0       0       0       0
     250       1     551     454     564     554       0       0       0       0
     251       1     454     536     545     564       0       0       0       0
     252       1     536     289     541     545       0       0       0       0
     253       1     545     544     566     564       0       0       0       0
     254       1     564     566     554     554       0       0       0       0
     255       1     566     544     567     557       0       0       0       0
     256       1     544     543     568     567       0       0       0       0
     257       1     557     567     568     559       0       0       0       0
     258       1     543     542     573     568       0       0       0       0
     259       1     559     568     573     561       0       0       0       0
     260       1     518     525     561     573       0       0       0       0
     261       1     573     519     518     518       0       0       0       0
     262       1     520     519     573     574       0       0       0       0
     263       1     521     520     574     575       0       0       0       0
     264       1     522     521     575     576       0       0       0       0
     265       1     515     522     576     577       0       0       0       0
     266       1     516     515     577     560       0       0       0       0
     267       1     573     542     540     574       0       0       0       0
     268       1     547     546     563     578       0       0       0       0
     269       1     563     562     552     578       0       0       0       0
     270       1     547     578     556     579       0       0       0       0
     271       1     579     548     547     547       0       0       0       0
     272       1     549     548     581     576       0       0       0       0
     273       1     540     549     576     575       0       0       0       0
     274       1     580     581     548     579       0       0       0       0
     275       1     540     575     574     574       0       0       0       0
     276       1     576     581     560     577       0       0       0       0
     277       1     558     560     581     580       0       0       0       0
     278       1     580     579     556     558       0       0       0       0
     279       1     507     494     597     597       0       0       0       0
     280       1     494     583     603     597       0       0       0       0
     281       1     589     596     600     311       0       0       0       0
     282       1     156     311     600     297       0       0       0       0
     283       1     596     595     601     600       0       0       0       0
     284       1     595     594     602     601       0       0       0       0
     285       1     594     593     603     602       0       0       0       0
     286       1     593     592     603     603       0       0       0       0
     287       1     597     603     592     506       0       0       0       0
     288       1     299     602     603     583       0       0       0       0
     289       1     299     587     601     602       0       0       0       0
     290       1     587     297     600     601       0       0       0       0
     291       1     607     609     610     608       0       0       0       0
     292       1     611     608     610     606       0       0       0       0
     293       1     361     606     610     610       0       0       0       0
     294       1     340     607     608     404       0       0       0       0
     295       1     639     404     608     642       0       0       0       0
     296       1     608     611     613     642       0       0       0       0
     297       1    1118     684     834     676       0       0       0       0
     298       1     684    1119     890     834       0       0       0       0
     299       1    1119     682     785     890       0       0       0       0
     300       1     682    1132     785     785       0       0       0       0
     301       1    1132    1134     787     785       0       0       0       0
     302       1    1134     687     848     787       0       0       0       0
     303       1     687    1131    1154     848       0       0       0       0
     304       1    1211     707     836     699       0       0       0       0
     305       1     707     706     910     836       0       0       0       0
     306       1     706     705     791     910       0       0       0       0
     307       1     705     708     791     791       0       0       0       0
     308       1     708    1227     793     791       0       0       0       0
     309       1    1227    1226     820     793       0       0       0       0
     310       1    1226     994    1002     820       0       0       0       0
     311       1     937     939     808     444       0       0       0       0
     312       1     939     637     796     808       0       0       0       0
     313       1     637     936     797     796       0       0       0       0
     314       1     936     639     797     797       0       0       0       0
     315       1     639     642     799     797       0       0       0       0
     316       1     642     613     861     799       0       0       0       0
     317       1     613     611     367     861       0       0       0       0
     318       1    1042    1039     672     821       0       0       0       0
     319       1    1043    1042     821     891       0       0       0       0
     320       1    1041    1043     891     804       0       0       0       0
     321       1    1024    1041     804     804       0       0       0       0
     322       1     660    1024     804     805       0       0       0       0
     323       1    1028     660     805     857       0       0       0       0
     324       1     952    1028     857     974       0       0       0       0
     325       1     443     444     808     807       0       0       0       0
     326       1     442     443     807     809       0       0       0       0
     327       1     441     442     809     810       0       0       0       0
     328       1     440     441     810     811       0       0       0       0
     329       1     630     440     811     812       0       0       0       0
     330       1     627     630     812     813       0       0       0       0
     331       1     813     862     997     996       0       0       0       0
     332       1     998     997     862     816       0       0       0       0
     333       1     999     998     816     817       0       0       0       0
     334       1     624     999     817     863       0       0       0       0
     335       1    1001     624     863     819       0       0       0       0
     336       1    1002    1001     819     820       0       0       0       0
     337       1     672     671     822     821       0       0       0       0
     338       1     671    1059     872     822       0       0       0       0
     339       1    1059     669     824     872       0       0       0       0
     340       1     669     668     825     824       0       0       0       0
     341       1     668     667     826     825       0       0       0       0
     342       1     667    1055     827     826       0       0       0       0
     343       1     829     827    1055     674       0       0       0       0
     344       1     681     680     830     829       0       0       0       0
     345       1     680     679     831     830       0       0       0       0
     346       1     679     678     832     831       0       0       0       0
     347       1     678     677     833     832       0       0       0       0
     348       1     677     676     834     833       0       0       0       0
     349       1     700     699     836     835       0       0       0       0
     350       1     701     700     835     837       0       0       0       0
     351       1     702     701     837     838       0       0       0       0
     352       1     703     702     838     839       0       0       0       0
     353       1    1202     703     839     840       0       0       0       0
     354       1     696    1202     840     841       0       0       0       0
     355       1     841     843    1149     689       0       0       0       0
     356       1    1150    1149     843     844       0       0       0       0
     357       1    1151    1150     844     845       0       0       0       0
     358       1     693    1151     845     846       0       0       0       0
     359       1    1153     693     846     847       0       0       0       0
     360       1    1154    1153     847     848       0       0       0       0
     361       1     849     366     651     650       0       0       0       0
     362       1     650     979     850     849       0       0       0       0
     363       1     372     366     849     851       0       0       0       0
     364       1     979     978     852     850       0       0       0       0
     365       1     648     372     851     853       0       0       0       0
     366       1     978     977     854     852       0       0       0       0
     367       1     977     655     855     854       0       0       0       0
     368       1     655     975     856     855       0       0       0       0
     369       1     975     974     857     856       0       0       0       0
     370       1     370     648     853     858       0       0       0       0
     371       1     369     370     858     859       0       0       0       0
     372       1     645     369     859     860       0       0       0       0
     373       1     367     645     860     861       0       0       0       0
     374       1     768     771     850     852       0       0       0       0
     375       1     849     850     771     851       0       0       0       0
     376       1     851     771     770     853       0       0       0       0
     377       1     853     770     769     858       0       0       0       0
     378       1     760     763     840     839       0       0       0       0
     379       1     753     751     816     862       0       0       0       0
     380       1     754     753     812     811       0       0       0       0
     381       1     753     862     813     812       0       0       0       0
     382       1     813     996     628     627       0       0       0       0
     383       1     754     811     810     755       0       0       0       0
     384       1     758     757     864     863       0       0       0       0
     385       1     751     758     863     817       0       0       0       0
     386       1     816     751     817     817       0       0       0       0
     387       1     765     764     866     865       0       0       0       0
     388       1     766     765     865     867       0       0       0       0
     389       1     775     824     825     825       0       0       0       0
     390       1     777     826     827     829       0       0       0       0
     391       1     778     777     829     830       0       0       0       0
     392       1     674     673     681     829       0       0       0       0
     393       1     778     830     831     779       0       0       0       0
     394       1     773     772     870     869       0       0       0       0
     395       1     774     773     869     871       0       0       0       0
     396       1     762     761     845     844       0       0       0       0
     397       1     763     762     844     843       0       0       0       0
     398       1     840     763     843     841       0       0       0       0
     399       1     841     689     697     696       0       0       0       0
     400       1     782     781     873     872       0       0       0       0
     401       1     775     782     872     824       0       0       0       0
     402       1     780     776     875     874       0       0       0       0
     403       1     781     780     874     876       0       0       0       0
     404       1     756     752     878     877       0       0       0       0
     405       1     757     756     877     879       0       0       0       0
     406       1     759     766     881     880       0       0       0       0
     407       1     761     759     880     882       0       0       0       0
     408       1     845     761     882     846       0       0       0       0
     409       1     883     752     755     810       0       0       0       0
     410       1     837     866     764     838       0       0       0       0
     411       1     760     839     838     764       0       0       0       0
     412       1     769     767     885     884       0       0       0       0
     413       1     737     736     874     875       0       0       0       0
     414       1     736     735     876     874       0       0       0       0
     415       1     738     737     875     886       0       0       0       0
     416       1     739     738     886     887       0       0       0       0
     417       1     735     734     923     876       0       0       0       0
     418       1     734     733     889     923       0       0       0       0
     419       1     740     739     887     890       0       0       0       0
     420       1     741     740     890     785       0       0       0       0
     421       1     733     732     891     889       0       0       0       0
     422       1     732     712     804     891       0       0       0       0
     423       1     781     876     923     873       0       0       0       0
     424       1     805     804     712     731       0       0       0       0
     425       1     787     742     741     785       0       0       0       0
     426       1     767     774     894     885       0       0       0       0
     427       1     805     731     730     896       0       0       0       0
     428       1     730     729     897     896       0       0       0       0
     429       1     729     728     869     897       0       0       0       0
     430       1     869     728     727     871       0       0       0       0
     431       1     871     727     894     774       0       0       0       0
     432       1     727     726     898     894       0       0       0       0
     433       1     726     725     899     898       0       0       0       0
     434       1     725     724     900     899       0       0       0       0
     435       1     724     723     799     900       0       0       0       0
     436       1     723     722     797     799       0       0       0       0
     437       1     796     797     722     721       0       0       0       0
     438       1     885     894     898     899       0       0       0       0
     439       1     787     903     743     742       0       0       0       0
     440       1     744     743     903     905       0       0       0       0
     441       1     745     744     905     906       0       0       0       0
     442       1     746     745     906     881       0       0       0       0
     443       1     746     867     747     747       0       0       0       0
     444       1     881     766     867     746       0       0       0       0
     445       1     747     867     865     907       0       0       0       0
     446       1     748     747     907     907       0       0       0       0
     447       1     749     748     907     908       0       0       0       0
     448       1     750     749     908     910       0       0       0       0
     449       1     711     750     910     791       0       0       0       0
     450       1     910     908     927     836       0       0       0       0
     451       1     793     713     711     791       0       0       0       0
     452       1     796     721     720     913       0       0       0       0
     453       1     720     719     914     913       0       0       0       0
     454       1     719     718     878     914       0       0       0       0
     455       1     718     717     877     878       0       0       0       0
     456       1     717     716     879     877       0       0       0       0
     457       1     716     715     915     879       0       0       0       0
     458       1     715     714     916     915       0       0       0       0
     459       1     793     916     714     713       0       0       0       0
     460       1     757     879     915     864       0       0       0       0
     461       1     881     906     905     880       0       0       0       0
     462       1     854     855     870     772       0       0       0       0
     463       1     854     772     768     852       0       0       0       0
     464       1     855     856     919     870       0       0       0       0
     465       1     869     870     919     897       0       0       0       0
     466       1     819     863     864     921       0       0       0       0
     467       1     769     884     859     858       0       0       0       0
     468       1     860     859     884     922       0       0       0       0
     469       1     884     885     899     922       0       0       0       0
     470       1     752     883     914     878       0       0       0       0
     471       1     873     923     821     822       0       0       0       0
     472       1     891     821     923     889       0       0       0       0
     473       1     820     819     921     924       0       0       0       0
     474       1     793     820     924     916       0       0       0       0
     475       1     810     809     807     883       0       0       0       0
     476       1     921     864     915     924       0       0       0       0
     477       1     776     779     925     925       0       0       0       0
     478       1     875     776     925     886       0       0       0       0
     479       1     865     866     926     907       0       0       0       0
     480       1     908     907     926     927       0       0       0       0
     481       1     861     860     922     928       0       0       0       0
     482       1     799     861     928     900       0       0       0       0
     483       1     856     857     929     919       0       0       0       0
     484       1     857     805     896     929       0       0       0       0
     485       1     808     796     913     930       0       0       0       0
     486       1     913     914     883     930       0       0       0       0
     487       1     787     848     932     903       0       0       0       0
     488       1     905     903     932     880       0       0       0       0
     489       1     848     847     933     932       0       0       0       0
     490       1     847     846     882     933       0       0       0       0
     491       1     807     808     930     883       0       0       0       0
     492       1     880     932     933     882       0       0       0       0
     493       1     928     922     899     900       0       0       0       0
     494       1     919     929     896     897       0       0       0       0
     495       1     822     872     873     873       0       0       0       0
     496       1     916     924     915     915       0       0       0       0
     497       1     835     926     866     837       0       0       0       0
     498       1     835     836     927     926       0       0       0       0
     499       1     779     831     832     925       0       0       0       0
     500       1     832     833     934     925       0       0       0       0
     501       1     833     834     935     934       0       0       0       0
     502       1     834     890     887     935       0       0       0       0
     503       1     935     887     886     886       0       0       0       0
     504       1     886     925     934     935       0       0       0       0
     505       1     825     826     777     775       0       0       0       0
     506       1     392     201     402     950       0       0       0       0
     507       1     402     936     637     950       0       0       0       0
     508       1     937     950     637     939       0       0       0       0
     509       1     207     392     950     407       0       0       0       0
     510       1     950     937     418     407       0       0       0       0
     511       1     418     416     407     407       0       0       0       0
     512       1     979     650     973     973       0       0       0       0
     513       1     979     973     971     971       0       0       0       0
     514       1     979     971     966     978       0       0       0       0
     515       1     966     967     977     978       0       0       0       0
     516       1     974     980     953     952       0       0       0       0
     517       1     980     981     951     953       0       0       0       0
     518       1     962     982     965     961       0       0       0       0
     519       1    1270     982     962    1268       0       0       0       0
     520       1     982     983     964     965       0       0       0       0
     521       1     983     984     963     964       0       0       0       0
     522       1     984     985     970     963       0       0       0       0
     523       1     985     981     969     970       0       0       0       0
     524       1     981     985     956     951       0       0       0       0
     525       1     985     984     954     956       0       0       0       0
     526       1     984     983    1269     954       0       0       0       0
     527       1     983     982    1270    1269       0       0       0       0
     528       1     967     968     655     977       0       0       0       0
     529       1     981     986     968     969       0       0       0       0
     530       1     968     986     975     655       0       0       0       0
     531       1     986     980     974     975       0       0       0       0
     532       1     981     980     986     986       0       0       0       0
     533       1     997    1004     996     996       0       0       0       0
     534       1    1004     997    1003    1003       0       0       0       0
     535       1     997     998    1005    1003       0       0       0       0
     536       1    1010    1005     998     999       0       0       0       0
     537       1    1015    1002     994     995       0       0       0       0
     538       1    1016    1015     995     991       0       0       0       0
     539       1    1017    1014    1294    1013       0       0       0       0
     540       1    1017     990     988    1014       0       0       0       0
     541       1    1018    1017    1013    1012       0       0       0       0
     542       1    1019    1018    1012    1006       0       0       0       0
     543       1    1020    1019    1006    1007       0       0       0       0
     544       1    1016    1020    1007    1008       0       0       0       0
     545       1    1020    1016     991     993       0       0       0       0
     546       1    1019    1020     993     987       0       0       0       0
     547       1    1018    1019     987     989       0       0       0       0
     548       1    1017    1018     989     990       0       0       0       0
     549       1    1009    1010     999     624       0       0       0       0
     550       1    1021    1016    1008    1009       0       0       0       0
     551       1    1021    1009     624    1001       0       0       0       0
     552       1    1015    1021    1001    1002       0       0       0       0
     553       1    1016    1021    1015    1015       0       0       0       0
     554       1    1286    1358    1036    1289       0       0       0       0
     555       1    1036    1024     660    1289       0       0       0       0
     556       1     952    1289     660    1028       0       0       0       0
     557       1    1041    1024    1036    1045       0       0       0       0
     558       1    1358    1031    1045    1036       0       0       0       0
     559       1    1031    1375    1040    1045       0       0       0       0
     560       1    1041    1045    1040    1043       0       0       0       0
     561       1    1040    1039    1042    1043       0       0       0       0
     562       1     667    1063    1055    1055       0       0       0       0
     563       1    1063     667    1062    1062       0       0       0       0
     564       1     667     668    1064    1062       0       0       0       0
     565       1    1069    1064     668     669       0       0       0       0
     566       1    1074     672    1039    1373       0       0       0       0
     567       1    1075    1074    1373    1050       0       0       0       0
     568       1    1076    1384    1070    1072       0       0       0       0
     569       1    1076    1430    1985    1384       0       0       0       0
     570       1    1077    1076    1072    1071       0       0       0       0
     571       1    1078    1077    1071    1065       0       0       0       0
     572       1    1079    1078    1065    1066       0       0       0       0
     573       1    1075    1079    1066    1067       0       0       0       0
     574       1    1079    1075    1050    1051       0       0       0       0
     575       1    1078    1079    1051    1415       0       0       0       0
     576       1    1077    1078    1415    1431       0       0       0       0
     577       1    1076    1077    1431    1430       0       0       0       0
     578       1    1068    1069     669    1059       0       0       0       0
     579       1    1080    1075    1067    1068       0       0       0       0
     580       1    1080    1068    1059     671       0       0       0       0
     581       1    1074    1080     671     672       0       0       0       0
     582       1    1075    1080    1074    1074       0       0       0       0
     583       1     681     673    1103    1103       0       0       0       0
     584       1     681    1103    1101    1101       0       0       0       0
     585       1     681    1101    1096     680       0       0       0       0
     586       1    1096    1097     679     680       0       0       0       0
     587       1     676    1110    1083    1118       0       0       0       0
     588       1    1110    1111    1488    1083       0       0       0       0
     589       1    1092    1112    1095    1091       0       0       0       0
     590       1    1468    1112    1092    1087       0       0       0       0
     591       1    1112    1113    1094    1095       0       0       0       0
     592       1    1113    1114    1093    1094       0       0       0       0
     593       1    1114    1115    1100    1093       0       0       0       0
     594       1    1115    1111    1099    1100       0       0       0       0
     595       1    1111    1115    1481    1488       0       0       0       0
     596       1    1115    1114    1467    1481       0       0       0       0
     597       1    1114    1113    1090    1467       0       0       0       0
     598       1    1113    1112    1468    1090       0       0       0       0
     599       1    1097    1098     678     679       0       0       0       0
     600       1    1111    1116    1098    1099       0       0       0       0
     601       1    1098    1116     677     678       0       0       0       0
     602       1    1116    1110     676     677       0       0       0       0
     603       1    1111    1110    1116    1116       0       0       0       0
     604       1    1493    1127    1124    1122       0       0       0       0
     605       1    1124     682    1119    1122       0       0       0       0
     606       1    1118    1122    1119     684       0       0       0       0
     607       1    1132     682    1124    1136       0       0       0       0
     608       1    1127    1135    1136    1124       0       0       0       0
     609       1    1135    1137    1512    1136       0       0       0       0
     610       1    1132    1136    1512    1134       0       0       0       0
     611       1    1512    1131     687    1134       0       0       0       0
     612       1    1149    1156     689     689       0       0       0       0
     613       1    1156    1149    1155    1155       0       0       0       0
     614       1    1149    1150    1157    1155       0       0       0       0
     615       1    1162    1157    1150    1151       0       0       0       0
     616       1    1167    1154    1131    1510       0       0       0       0
     617       1    1168    1167    1510    1553       0       0       0       0
     618       1    1169    1166    1163    1165       0       0       0       0
     619       1    1169    1566    1564    1166       0       0       0       0
     620       1    1170    1169    1165    1164       0       0       0       0
     621       1    1171    1170    1164    1158       0       0       0       0
     622       1    1172    1171    1158    1159       0       0       0       0
     623       1    1168    1172    1159    1160       0       0       0       0
     624       1    1172    1168    1553    1145       0       0       0       0
     625       1    1171    1172    1145    1563       0       0       0       0
     626       1    1170    1171    1563    1141       0       0       0       0
     627       1    1169    1170    1141    1566       0       0       0       0
     628       1    1161    1162    1151     693       0       0       0       0
     629       1    1173    1168    1160    1161       0       0       0       0
     630       1    1173    1161     693    1153       0       0       0       0
     631       1    1167    1173    1153    1154       0       0       0       0
     632       1    1168    1173    1167    1167       0       0       0       0
     633       1    1202     696    1196    1196       0       0       0       0
     634       1    1202    1196    1194    1194       0       0       0       0
     635       1    1202    1194    1189     703       0       0       0       0
     636       1    1189    1190     702     703       0       0       0       0
     637       1     699    1203    1631    1211       0       0       0       0
     638       1    1203    1204    1618    1631       0       0       0       0
     639       1    1588    1205    1188    1578       0       0       0       0
     640       1    1182    1205    1588    1180       0       0       0       0
     641       1    1205    1206    1187    1188       0       0       0       0
     642       1    1206    1207    1186    1187       0       0       0       0
     643       1    1207    1208    1193    1186       0       0       0       0
     644       1    1208    1204    1192    1193       0       0       0       0
     645       1    1204    1208    1623    1618       0       0       0       0
     646       1    1208    1207    1621    1623       0       0       0       0
     647       1    1207    1206    1183    1621       0       0       0       0
     648       1    1206    1205    1182    1183       0       0       0       0
     649       1    1190    1191     701     702       0       0       0       0
     650       1    1204    1209    1191    1192       0       0       0       0
     651       1    1191    1209     700     701       0       0       0       0
     652       1    1209    1203     699     700       0       0       0       0
     653       1    1204    1203    1209    1209       0       0       0       0
     654       1    1636    1216    1217    1635       0       0       0       0
     655       1    1217     705     706    1635       0       0       0       0
     656       1    1211    1635     706     707       0       0       0       0
     657       1     708     705    1217    1229       0       0       0       0
     658       1    1216    1228    1229    1217       0       0       0       0
     659       1    1228    1327    1328    1229       0       0       0       0
     660       1     708    1229    1328    1227       0       0       0       0
     661       1    1328     994    1226    1227       0       0       0       0
     662       1    1772    1254    1738    1733       0       0       0       0
     663       1    1254    1243    1233    1738       0       0       0       0
     664       1    1236    1255     962     961       0       0       0       0
     665       1    1255    1246    1268     962       0       0       0       0
     666       1    1256    1254    1772    1771       0       0       0       0
     667       1    1257    1255    1236    1237       0       0       0       0
     668       1    1258    1257    1237    1238       0       0       0       0
     669       1    1259    1258    1238    1239       0       0       0       0
     670       1    1260    1256    1771    1770       0       0       0       0
     671       1    1261    1259    1239    1240       0       0       0       0
     672       1    1261    1260    1770    1769       0       0       0       0
     673       1    1259    1261    1769    1249       0       0       0       0
     674       1    1258    1259    1249    1767       0       0       0       0
     675       1    1257    1258    1767    1247       0       0       0       0
     676       1    1255    1257    1247    1246       0       0       0       0
     677       1    1260    1261    1240    1241       0       0       0       0
     678       1    1256    1260    1241    1242       0       0       0       0
     679       1    1254    1256    1242    1243       0       0       0       0
     680       1    1267     954    1269    1272       0       0       0       0
     681       1    1263    1267    1272    1779       0       0       0       0
     682       1    1268    1763    1272    1270       0       0       0       0
     683       1    1269    1270    1272    1272       0       0       0       0
     684       1    1763    1776    1778    1272       0       0       0       0
     685       1    1272    1778    1779    1779       0       0       0       0
     686       1    1293     951     956    1285       0       0       0       0
     687       1    1273    1293    1285    1275       0       0       0       0
     688       1     956     954    1267    1285       0       0       0       0
     689       1    1267    1263    1275    1285       0       0       0       0
     690       1    1273    1286    1289    1293       0       0       0       0
     691       1    1289     952     953    1293       0       0       0       0
     692       1     953     951    1293    1293       0       0       0       0
     693       1    1315    1316    1014     988       0       0       0       0
     694       1    1316    1298    1294    1014       0       0       0       0
     695       1    1305    1317    1809    1297       0       0       0       0
     696       1    1317    1308    1793    1809       0       0       0       0
     697       1    1318    1317    1305    1304       0       0       0       0
     698       1    1319    1316    1315    1314       0       0       0       0
     699       1    1320    1319    1314    1313       0       0       0       0
     700       1    1321    1320    1313    1312       0       0       0       0
     701       1    1322    1318    1304    1303       0       0       0       0
     702       1    1323    1321    1312    1311       0       0       0       0
     703       1    1323    1322    1303    1302       0       0       0       0
     704       1    1321    1323    1302    1301       0       0       0       0
     705       1    1320    1321    1301    1300       0       0       0       0
     706       1    1319    1320    1300    1299       0       0       0       0
     707       1    1316    1319    1299    1298       0       0       0       0
     708       1    1322    1323    1311    1310       0       0       0       0
     709       1    1318    1322    1310    1309       0       0       0       0
     710       1    1317    1318    1309    1308       0       0       0       0
     711       1    1327    1329    1330    1328       0       0       0       0
     712       1     994    1328    1330     995       0       0       0       0
     713       1     991     995    1330    1330       0       0       0       0
     714       1    1333     987     993    1343       0       0       0       0
     715       1    1348    1333    1343    1342       0       0       0       0
     716       1     993     991    1330    1343       0       0       0       0
     717       1    1330    1329    1342    1343       0       0       0       0
     718       1     987    1333    1355     989       0       0       0       0
     719       1    1333    1348    1353    1355       0       0       0       0
     720       1    1827     988     990    1355       0       0       0       0
     721       1     990     989    1355    1355       0       0       0       0
     722       1    1350    1827    1355    1352       0       0       0       0
     723       1    1352    1355    1353    1353       0       0       0       0
     724       1    1286    1360    1358    1358       0       0       0       0
     725       1    1360    1286    1273    1787       0       0       0       0
     726       1    1360    1381    1031    1358       0       0       0       0
     727       1    1381    1360    1787    1938       0       0       0       0
     728       1    1375    1413    1371    1040       0       0       0       0
     729       1    1039    1040    1371    1373       0       0       0       0
     730       1    1050    1373    1371    1371       0       0       0       0
     731       1    1375    1031    1381    1381       0       0       0       0
     732       1    1375    1381    1938    1413       0       0       0       0
     733       1    1993    1404    1384    1985       0       0       0       0
     734       1    1404    1386    1070    1384       0       0       0       0
     735       1    1393    1405    1966    1960       0       0       0       0
     736       1    1405    1986    1983    1966       0       0       0       0
     737       1    1406    1405    1393    1392       0       0       0       0
     738       1    1407    1404    1993    1992       0       0       0       0
     739       1    1408    1407    1992    1991       0       0       0       0
     740       1    1409    1408    1991    1990       0       0       0       0
     741       1    1410    1406    1392    1391       0       0       0       0
     742       1    1411    1409    1990    1989       0       0       0       0
     743       1    1411    1410    1391    1390       0       0       0       0
     744       1    1409    1411    1390    1389       0       0       0       0
     745       1    1408    1409    1389    1388       0       0       0       0
     746       1    1407    1408    1388    1387       0       0       0       0
     747       1    1404    1407    1387    1386       0       0       0       0
     748       1    1410    1411    1989    1988       0       0       0       0
     749       1    1406    1410    1988    1987       0       0       0       0
     750       1    1405    1406    1987    1986       0       0       0       0
     751       1    1371    1413    1944    1422       0       0       0       0
     752       1    1050    1371    1422    1051       0       0       0       0
     753       1    1944    1942    1416    1422       0       0       0       0
     754       1    1416    1415    1051    1422       0       0       0       0
     755       1    1415    1416    1435    1431       0       0       0       0
     756       1    1416    1942    2000    1435       0       0       0       0
     757       1    1997    1428    1435    1999       0       0       0       0
     758       1    2000    1999    1435    1435       0       0       0       0
     759       1    1428    1985    1430    1435       0       0       0       0
     760       1    1430    1431    1435    1435       0       0       0       0
     761       1    2113    1458    2014    1436       0       0       0       0
     762       1    1458    1447    2013    2014       0       0       0       0
     763       1    1440    1459    1092    1091       0       0       0       0
     764       1    1459    2106    1087    1092       0       0       0       0
     765       1    1460    1458    2113    1456       0       0       0       0
     766       1    1461    1459    1440    1441       0       0       0       0
     767       1    1462    1461    1441    1442       0       0       0       0
     768       1    1463    1462    1442    1443       0       0       0       0
     769       1    1464    1460    1456    1455       0       0       0       0
     770       1    1465    1463    1443    1444       0       0       0       0
     771       1    1465    1464    1455    1454       0       0       0       0
     772       1    1463    1465    1454    2109       0       0       0       0
     773       1    1462    1463    2109    1452       0       0       0       0
     774       1    1461    1462    1452    2107       0       0       0       0
     775       1    1459    1461    2107    2106       0       0       0       0
     776       1    1464    1465    1444    1445       0       0       0       0
     777       1    1460    1464    1445    1446       0       0       0       0
     778       1    1458    1460    1446    1447       0       0       0       0
     779       1    1477    1467    1090    1478       0       0       0       0
     780       1    1472    1477    1478    2131       0       0       0       0
     781       1    1087    1471    1478    1468       0       0       0       0
     782       1    1090    1468    1478    1478       0       0       0       0
     783       1    1471    2128    2130    1478       0       0       0       0
     784       1    1478    2130    2131    2131       0       0       0       0
     785       1    1486    1488    1481    1487       0       0       0       0
     786       1    1491    1486    1487    2139       0       0       0       0
     787       1    1481    1467    1477    1487       0       0       0       0
     788       1    1477    1472    2139    1487       0       0       0       0
     789       1    1491    1493    1122    1486       0       0       0       0
     790       1    1122    1118    1083    1486       0       0       0       0
     791       1    1083    1488    1486    1486       0       0       0       0
     792       1    1493    1501    1127    1127       0       0       0       0
     793       1    1501    1493    1491    2147       0       0       0       0
     794       1    1501    1505    1135    1127       0       0       0       0
     795       1    1505    1501    2147    2160       0       0       0       0
     796       1    1137    1556    1514    1512       0       0       0       0
     797       1    1131    1512    1514    1510       0       0       0       0
     798       1    1553    1510    1514    1514       0       0       0       0
     799       1    1137    1135    1505    1505       0       0       0       0
     800       1    1137    1505    2160    1556       0       0       0       0
     801       1    1541    1542    1166    1564       0       0       0       0
     802       1    1542    1524    1163    1166       0       0       0       0
     803       1    1531    1543    1533    1523       0       0       0       0
     804       1    1543    1534    1532    1533       0       0       0       0
     805       1    1544    1543    1531    1530       0       0       0       0
     806       1    1545    1542    1541    1540       0       0       0       0
     807       1    1546    1545    1540    2194       0       0       0       0
     808       1    1547    1546    2194    2193       0       0       0       0
     809       1    1548    1544    1530    1529       0       0       0       0
     810       1    1549    1547    2193    2192       0       0       0       0
     811       1    1549    1548    1529    1528       0       0       0       0
     812       1    1547    1549    1528    1527       0       0       0       0
     813       1    1546    1547    1527    1526       0       0       0       0
     814       1    1545    1546    1526    1525       0       0       0       0
     815       1    1542    1545    1525    1524       0       0       0       0
     816       1    1548    1549    2192    1536       0       0       0       0
     817       1    1544    1548    1536    2190       0       0       0       0
     818       1    1543    1544    2190    1534       0       0       0       0
     819       1    1568    1563    1145    1562       0       0       0       0
     820       1    1550    1568    1562    2157       0       0       0       0
     821       1    1145    1553    1514    1562       0       0       0       0
     822       1    1514    1556    2157    1562       0       0       0       0
     823       1    1563    1568    1574    1141       0       0       0       0
     824       1    1568    1550    1572    1574       0       0       0       0
     825       1    2198    1564    1566    1574       0       0       0       0
     826       1    1566    1141    1574    1574       0       0       0       0
     827       1    2211    2198    1574    2213       0       0       0       0
     828       1    2213    1574    1572    1572       0       0       0       0
     829       1    2251    1597    2228    2243       0       0       0       0
     830       1    1597    1586    2227    2228       0       0       0       0
     831       1    1579    1598    1588    1578       0       0       0       0
     832       1    1598    2244    1180    1588       0       0       0       0
     833       1    1599    1597    2251    2250       0       0       0       0
     834       1    1600    1598    1579    1580       0       0       0       0
     835       1    1601    1600    1580    1581       0       0       0       0
     836       1    1602    1601    1581    1582       0       0       0       0
     837       1    1603    1599    2250    2249       0       0       0       0
     838       1    1604    1602    1582    1583       0       0       0       0
     839       1    1604    1603    2249    2248       0       0       0       0
     840       1    1602    1604    2248    2247       0       0       0       0
     841       1    1601    1602    2247    2246       0       0       0       0
     842       1    1600    1601    2246    2245       0       0       0       0
     843       1    1598    1600    2245    2244       0       0       0       0
     844       1    1603    1604    1583    1584       0       0       0       0
     845       1    1599    1603    1584    1585       0       0       0       0
     846       1    1597    1599    1585    1586       0       0       0       0
     847       1    1625    1621    1183    1617       0       0       0       0
     848       1    1611    1625    1617    2269       0       0       0       0
     849       1    1180    1610    1617    1182       0       0       0       0
     850       1    1183    1182    1617    1617       0       0       0       0
     851       1    1610    2619    2268    1617       0       0       0       0
     852       1    1617    2268    2269    2269       0       0       0       0
     853       1    1632    1633    1628    1627       0       0       0       0
     854       1    1633    1618    1623    1628       0       0       0       0
     855       1    1611    1627    1628    1625       0       0       0       0
     856       1    1621    1625    1628    1623       0       0       0       0
     857       1    1632    1636    1635    1633       0       0       0       0
     858       1    1635    1211    1631    1633       0       0       0       0
     859       1    1631    1618    1633    1633       0       0       0       0
     860       1    1636    1648    1216    1216       0       0       0       0
     861       1    1648    1636    1632    1647       0       0       0       0
     862       1    1648    1646    1228    1216       0       0       0       0
     863       1    1646    1648    1647    1921       0       0       0       0
     864       1    1327    1228    1646    1646       0       0       0       0
     865       1    1327    1646    1921    1329       0       0       0       0
     866       1    1654    1658    1698    1745       0       0       0       0
     867       1    1731    1745    1698    2313       0       0       0       0
     868       1    1679    1958    1681    1699       0       0       0       0
     869       1    1681    1680    1684    1699       0       0       0       0
     870       1    1658    1659    1700    1698       0       0       0       0
     871       1    1659    1660    1714    1700       0       0       0       0
     872       1    1678    1679    1699    1702       0       0       0       0
     873       1    1677    1678    1702    1703       0       0       0       0
     874       1    1660    1661    1704    1714       0       0       0       0
     875       1    1676    1677    1703    1705       0       0       0       0
     876       1    1661    1662    1706    1704       0       0       0       0
     877       1    1675    1676    1705    1707       0       0       0       0
     878       1    1662    1657    1708    1706       0       0       0       0
     879       1    1657    1664    1665    1708       0       0       0       0
     880       1    1671    1675    1707    1709       0       0       0       0
     881       1    1672    1671    1709    1673       0       0       0       0
     882       1    1699    1684    1710    1702       0       0       0       0
     883       1    1684    1683    1711    1710       0       0       0       0
     884       1    1683    1682    1693    1711       0       0       0       0
     885       1    1693    1682    2311    1689       0       0       0       0
     886       1    2313    1698    1700    1712       0       0       0       0
     887       1    2312    2313    1712    1694       0       0       0       0
     888       1    2311    2312    1694    1689       0       0       0       0
     889       1    1714    1695    1694    1712       0       0       0       0
     890       1    1712    1700    1714    1714       0       0       0       0
     891       1    1695    1714    1704    1715       0       0       0       0
     892       1    1696    1695    1715    1716       0       0       0       0
     893       1    1715    1704    1706    1716       0       0       0       0
     894       1    1697    1696    1716    1721       0       0       0       0
     895       1    1716    1706    1708    1721       0       0       0       0
     896       1    1663    1670    1721    1721       0       0       0       0
     897       1    1721    1708    1665    1663       0       0       0       0
     898       1    1670    1669    1722    1721       0       0       0       0
     899       1    1669    1668    1723    1722       0       0       0       0
     900       1    1668    1667    1724    1723       0       0       0       0
     901       1    1667    1666    1725    1724       0       0       0       0
     902       1    1666    1673    1709    1725       0       0       0       0
     903       1    1697    1721    1722    1688       0       0       0       0
     904       1    1690    1688    1723    1724       0       0       0       0
     905       1    1723    1688    1722    1722       0       0       0       0
     906       1    1725    1709    1726    1724       0       0       0       0
     907       1    1690    1724    1726    1691       0       0       0       0
     908       1    1726    1709    1707    1705       0       0       0       0
     909       1    1693    1692    1728    1711       0       0       0       0
     910       1    1710    1711    1728    1728       0       0       0       0
     911       1    1692    1691    1729    1728       0       0       0       0
     912       1    1705    1729    1691    1726       0       0       0       0
     913       1    1705    1703    1728    1729       0       0       0       0
     914       1    1728    1703    1702    1710       0       0       0       0
     915       1    1731    2335    1745    1745       0       0       0       0
     916       1    2335    2322    1751    1745       0       0       0       0
     917       1    1233    1740    1748    1738       0       0       0       0
     918       1    1733    1738    1748    1734       0       0       0       0
     919       1    1740    1741    1749    1748       0       0       0       0
     920       1    1741    1742    1750    1749       0       0       0       0
     921       1    1742    1743    1751    1750       0       0       0       0
     922       1    1743    1744    1751    1751       0       0       0       0
     923       1    1745    1751    1744    1654       0       0       0       0
     924       1    1736    1750    1751    2322       0       0       0       0
     925       1    1736    1735    1749    1750       0       0       0       0
     926       1    1735    1734    1748    1749       0       0       0       0
     927       1    2340    1776    1763    1246       0       0       0       0
     928       1    1763    1268    1246    1246       0       0       0       0
     929       1    1733    1773    2347    1772       0       0       0       0
     930       1    1773    2339    2347    2347       0       0       0       0
     931       1    1771    1772    2347    2346       0       0       0       0
     932       1    1770    1771    2346    1759       0       0       0       0
     933       1    1759    2344    1769    1770       0       0       0       0
     934       1    2344    1757    1249    1769       0       0       0       0
     935       1    1767    1249    1757    1756       0       0       0       0
     936       1    1756    1755    1247    1767       0       0       0       0
     937       1    2340    1246    1247    1755       0       0       0       0
     938       1    1779    2360    1788    1263       0       0       0       0
     939       1    1783    1778    1776    2670       0       0       0       0
     940       1    1778    1783    2360    1779       0       0       0       0
     941       1    1275    1789    1787    1273       0       0       0       0
     942       1    1790    1275    1263    1788       0       0       0       0
     943       1    1789    1275    1790    1790       0       0       0       0
     944       1    2372    2409    1802    1802       0       0       0       0
     945       1    2374    2372    1802    1811       0       0       0       0
     946       1    1808    1297    1809    1812       0       0       0       0
     947       1    1809    1793    1795    1812       0       0       0       0
     948       1    1807    1808    1812    1813       0       0       0       0
     949       1    1806    1807    1813    1814       0       0       0       0
     950       1    1805    1806    1814    1811       0       0       0       0
     951       1    1804    1805    1811    1811       0       0       0       0
     952       1    1804    1811    1802    1801       0       0       0       0
     953       1    1811    1814    2377    2374       0       0       0       0
     954       1    1796    2377    1814    1813       0       0       0       0
     955       1    1795    1796    1813    1812       0       0       0       0
     956       1    1827    1350    2398    1315       0       0       0       0
     957       1    1827    1315     988     988       0       0       0       0
     958       1    2379    1829    2696    2696       0       0       0       0
     959       1    2379    1793    1308    1829       0       0       0       0
     960       1    1830    1829    1308    1309       0       0       0       0
     961       1    1831    1830    1309    1310       0       0       0       0
     962       1    1832    1831    1310    1311       0       0       0       0
     963       1    1833    1832    1311    1312       0       0       0       0
     964       1    1312    1313    2396    1833       0       0       0       0
     965       1    2397    2396    1313    1314       0       0       0       0
     966       1    1314    1315    2398    2397       0       0       0       0
     967       1    2229    1844    1884    1842       0       0       0       0
     968       1    1841    1842    1884    1872       0       0       0       0
     969       1    1865    1801    1802    1885       0       0       0       0
     970       1    1802    2409    2411    1885       0       0       0       0
     971       1    1844    1845    1886    1884       0       0       0       0
     972       1    1845    1846    1912    1886       0       0       0       0
     973       1    1864    1865    1885    1888       0       0       0       0
     974       1    1863    1864    1888    1900       0       0       0       0
     975       1    1846    1847    1890    1912       0       0       0       0
     976       1    1862    1863    1900    1891       0       0       0       0
     977       1    1847    1848    1892    1890       0       0       0       0
     978       1    1861    1862    1891    1893       0       0       0       0
     979       1    1848    1843    1894    1892       0       0       0       0
     980       1    1843    1851    1850    1894       0       0       0       0
     981       1    1857    1861    1893    1895       0       0       0       0
     982       1    1858    1857    1895    1859       0       0       0       0
     983       1    1872    1884    1886    1896       0       0       0       0
     984       1    1873    1872    1896    1897       0       0       0       0
     985       1    1871    1873    1897    1880       0       0       0       0
     986       1    1871    1880    1875    2410       0       0       0       0
     987       1    1885    2411    1898    1888       0       0       0       0
     988       1    2411    2412    1879    1898       0       0       0       0
     989       1    2412    2410    1875    1879       0       0       0       0
     990       1    1879    1878    1900    1898       0       0       0       0
     991       1    1898    1900    1888    1888       0       0       0       0
     992       1    1900    1878    1901    1891       0       0       0       0
     993       1    1878    1877    1902    1901       0       0       0       0
     994       1    1891    1901    1902    1893       0       0       0       0
     995       1    1877    1876    1907    1902       0       0       0       0
     996       1    1893    1902    1907    1895       0       0       0       0
     997       1    1852    1859    1895    1907       0       0       0       0
     998       1    1907    1853    1852    1852       0       0       0       0
     999       1    1854    1853    1907    1908       0       0       0       0
    1000       1    1855    1854    1908    1909       0       0       0       0
    1001       1    1856    1855    1909    1910       0       0       0       0
    1002       1    1849    1856    1910    1911       0       0       0       0
    1003       1    1850    1849    1911    1894       0       0       0       0
    1004       1    1907    1876    1874    1908       0       0       0       0
    1005       1    1881    1880    1897    1912       0       0       0       0
    1006       1    1897    1896    1886    1912       0       0       0       0
    1007       1    1881    1912    1890    1913       0       0       0       0
    1008       1    1913    1882    1881    1881       0       0       0       0
    1009       1    1883    1882    1915    1910       0       0       0       0
    1010       1    1874    1883    1910    1909       0       0       0       0
    1011       1    1914    1915    1882    1913       0       0       0       0
    1012       1    1874    1909    1908    1908       0       0       0       0
    1013       1    1910    1915    1894    1911       0       0       0       0
    1014       1    1892    1894    1915    1914       0       0       0       0
    1015       1    1914    1913    1890    1892       0       0       0       0
    1016       1    1923    1342    1329    1921       0       0       0       0
    1017       1    1342    1924    1919    1348       0       0       0       0
    1018       1    1923    1924    1342    1342       0       0       0       0
    1019       1    1352    1931    2706    1350       0       0       0       0
    1020       1    1932    1353    1348    1919       0       0       0       0
    1021       1    1353    1932    1931    1352       0       0       0       0
    1022       1    1789    1940    1938    1787       0       0       0       0
    1023       1    1941    1790    1788    1939       0       0       0       0
    1024       1    1790    1941    1940    1789       0       0       0       0
    1025       1    1940    1944    1413    1938       0       0       0       0
    1026       1    1944    1941    1939    1942       0       0       0       0
    1027       1    1940    1941    1944    1944       0       0       0       0
    1028       1    2445    1680    1681    1681       0       0       0       0
    1029       1    1952    2445    1681    1968       0       0       0       0
    1030       1    1965    1960    1966    1969       0       0       0       0
    1031       1    1966    1983    1953    1969       0       0       0       0
    1032       1    1964    1965    1969    1970       0       0       0       0
    1033       1    1963    1964    1970    1971       0       0       0       0
    1034       1    1962    1963    1971    1968       0       0       0       0
    1035       1    1961    1962    1968    1968       0       0       0       0
    1036       1    1961    1968    1681    1958       0       0       0       0
    1037       1    1968    1971    1955    1952       0       0       0       0
    1038       1    1954    1955    1971    1970       0       0       0       0
    1039       1    1953    1954    1970    1969       0       0       0       0
    1040       1    1428    1997    1982    1993       0       0       0       0
    1041       1    1428    1993    1985    1985       0       0       0       0
    1042       1    2459    1983    1986    2464       0       0       0       0
    1043       1    2459    2464    1973    1973       0       0       0       0
    1044       1    1987    1988    2466    2465       0       0       0       0
    1045       1    2467    2466    1988    1989       0       0       0       0
    1046       1    2468    2467    1989    1990       0       0       0       0
    1047       1    1990    1991    2469    2468       0       0       0       0
    1048       1    2470    2469    1991    1992       0       0       0       0
    1049       1    2465    2464    1986    1987       0       0       0       0
    1050       1    1992    1993    1982    2470       0       0       0       0
    1051       1    1999    2003    2669    1997       0       0       0       0
    1052       1    2004    2000    1942    1939       0       0       0       0
    1053       1    2000    2004    2003    1999       0       0       0       0
    1054       1    2518    2008    2030    2030       0       0       0       0
    1055       1    2008    2483    2027    2030       0       0       0       0
    1056       1    2013    2016    2024    2014       0       0       0       0
    1057       1    1436    2014    2024    2484       0       0       0       0
    1058       1    2016    2017    2025    2024       0       0       0       0
    1059       1    2017    2018    2026    2025       0       0       0       0
    1060       1    2018    2019    2027    2026       0       0       0       0
    1061       1    2019    2020    2027    2027       0       0       0       0
    1062       1    2030    2027    2020    2029       0       0       0       0
    1063       1    2486    2026    2027    2483       0       0       0       0
    1064       1    2486    2011    2025    2026       0       0       0       0
    1065       1    2011    2484    2024    2025       0       0       0       0
    1066       1    2029    2032    2072    2030       0       0       0       0
    1067       1    2518    2030    2072    2060       0       0       0       0
    1068       1    2053    2172    2055    2073       0       0       0       0
    1069       1    2055    2054    2057    2073       0       0       0       0
    1070       1    2032    2033    2074    2072       0       0       0       0
    1071       1    2033    2034    2100    2074       0       0       0       0
    1072       1    2052    2053    2073    2076       0       0       0       0
    1073       1    2051    2052    2076    2088       0       0       0       0
    1074       1    2034    2035    2078    2100       0       0       0       0
    1075       1    2050    2051    2088    2079       0       0       0       0
    1076       1    2035    2036    2080    2078       0       0       0       0
    1077       1    2049    2050    2079    2081       0       0       0       0
    1078       1    2036    2031    2082    2080       0       0       0       0
    1079       1    2031    2039    2038    2082       0       0       0       0
    1080       1    2045    2049    2081    2083       0       0       0       0
    1081       1    2046    2045    2083    2047       0       0       0       0
    1082       1    2060    2072    2074    2084       0       0       0       0
    1083       1    2061    2060    2084    2085       0       0       0       0
    1084       1    2519    2061    2085    2068       0       0       0       0
    1085       1    2519    2068    2063    2513       0       0       0       0
    1086       1    2073    2057    2086    2076       0       0       0       0
    1087       1    2057    2058    2067    2086       0       0       0       0
    1088       1    2058    2513    2063    2067       0       0       0       0
    1089       1    2067    2066    2088    2086       0       0       0       0
    1090       1    2086    2088    2076    2076       0       0       0       0
    1091       1    2088    2066    2089    2079       0       0       0       0
    1092       1    2066    2065    2090    2089       0       0       0       0
    1093       1    2079    2089    2090    2081       0       0       0       0
    1094       1    2065    2064    2095    2090       0       0       0       0
    1095       1    2081    2090    2095    2083       0       0       0       0
    1096       1    2040    2047    2083    2095       0       0       0       0
    1097       1    2095    2041    2040    2040       0       0       0       0
    1098       1    2042    2041    2095    2096       0       0       0       0
    1099       1    2043    2042    2096    2097       0       0       0       0
    1100       1    2044    2043    2097    2098       0       0       0       0
    1101       1    2037    2044    2098    2099       0       0       0       0
    1102       1    2038    2037    2099    2082       0       0       0       0
    1103       1    2095    2064    2062    2096       0       0       0       0
    1104       1    2069    2068    2085    2100       0       0       0       0
    1105       1    2085    2084    2074    2100       0       0       0       0
    1106       1    2069    2100    2078    2101       0       0       0       0
    1107       1    2101    2070    2069    2069       0       0       0       0
    1108       1    2071    2070    2103    2098       0       0       0       0
    1109       1    2062    2071    2098    2097       0       0       0       0
    1110       1    2102    2103    2070    2101       0       0       0       0
    1111       1    2062    2097    2096    2096       0       0       0       0
    1112       1    2098    2103    2082    2099       0       0       0       0
    1113       1    2080    2082    2103    2102       0       0       0       0
    1114       1    2102    2101    2078    2080       0       0       0       0
    1115       1    2528    2128    1471    2106       0       0       0       0
    1116       1    1471    1087    2106    2106       0       0       0       0
    1117       1    1436    2493    2124    2113       0       0       0       0
    1118       1    2493    2114    2124    2124       0       0       0       0
    1119       1    1456    2113    2124    2534       0       0       0       0
    1120       1    1455    1456    2534    2533       0       0       0       0
    1121       1    2533    2532    1454    1455       0       0       0       0
    1122       1    2532    2120    2109    1454       0       0       0       0
    1123       1    1452    2109    2120    2530       0       0       0       0
    1124       1    2530    2118    2107    1452       0       0       0       0
    1125       1    2106    2107    2118    2528       0       0       0       0
    1126       1    2131    2135    2148    1472       0       0       0       0
    1127       1    2136    2130    2128    2547       0       0       0       0
    1128       1    2130    2136    2135    2131       0       0       0       0
    1129       1    2139    2149    2147    1491       0       0       0       0
    1130       1    2144    2139    1472    2148       0       0       0       0
    1131       1    2149    2139    2144    2144       0       0       0       0
    1132       1    2149    2153    2160    2147       0       0       0       0
    1133       1    2154    2144    2148    2158       0       0       0       0
    1134       1    2144    2154    2153    2149       0       0       0       0
    1135       1    2153    2157    1556    2160       0       0       0       0
    1136       1    2157    2154    2158    1550       0       0       0       0
    1137       1    2153    2154    2157    2157       0       0       0       0
    1138       1    2560    2054    2055    2055       0       0       0       0
    1139       1    2165    2560    2055    2182       0       0       0       0
    1140       1    2179    1523    1533    2183       0       0       0       0
    1141       1    1533    1532    2166    2183       0       0       0       0
    1142       1    2178    2179    2183    2184       0       0       0       0
    1143       1    2177    2178    2184    2185       0       0       0       0
    1144       1    2176    2177    2185    2182       0       0       0       0
    1145       1    2175    2176    2182    2182       0       0       0       0
    1146       1    2175    2182    2055    2172       0       0       0       0
    1147       1    2182    2185    2565    2165       0       0       0       0
    1148       1    2167    2565    2185    2184       0       0       0       0
    1149       1    2166    2167    2184    2183       0       0       0       0
    1150       1    2198    2211    2586    1541       0       0       0       0
    1151       1    2198    1541    1564    1564       0       0       0       0
    1152       1    2208    2579    2799    2799       0       0       0       0
    1153       1    2208    1532    1534    2579       0       0       0       0
    1154       1    2201    2579    1534    2190       0       0       0       0
    1155       1    2581    2201    2190    1536       0       0       0       0
    1156       1    2203    2581    1536    2192       0       0       0       0
    1157       1    2204    2203    2192    2193       0       0       0       0
    1158       1    2193    2194    2205    2204       0       0       0       0
    1159       1    2206    2205    2194    1540       0       0       0       0
    1160       1    1540    1541    2586    2206       0       0       0       0
    1161       1    2213    2552    2551    2211       0       0       0       0
    1162       1    2218    1572    1550    2158       0       0       0       0
    1163       1    1572    2218    2552    2213       0       0       0       0
    1164       1    1841    2222    1842    1842       0       0       0       0
    1165       1    2222    2221    2241    1842       0       0       0       0
    1166       1    2227    2230    2238    2228       0       0       0       0
    1167       1    2243    2228    2238    2599       0       0       0       0
    1168       1    2230    2231    2239    2238       0       0       0       0
    1169       1    2231    2232    2240    2239       0       0       0       0
    1170       1    2232    2233    2241    2240       0       0       0       0
    1171       1    2233    2234    2241    2241       0       0       0       0
    1172       1    1842    2241    2234    2229       0       0       0       0
    1173       1    2601    2240    2241    2221       0       0       0       0
    1174       1    2601    2600    2239    2240       0       0       0       0
    1175       1    2600    2599    2238    2239       0       0       0       0
    1176       1    2621    2619    1610    2244       0       0       0       0
    1177       1    1610    1180    2244    2244       0       0       0       0
    1178       1    2243    2253    2628    2251       0       0       0       0
    1179       1    2253    2809    2628    2628       0       0       0       0
    1180       1    2250    2251    2628    2627       0       0       0       0
    1181       1    2249    2250    2627    2626       0       0       0       0
    1182       1    2626    2625    2248    2249       0       0       0       0
    1183       1    2625    2624    2247    2248       0       0       0       0
    1184       1    2246    2247    2624    2623       0       0       0       0
    1185       1    2623    2622    2245    2246       0       0       0       0
    1186       1    2244    2245    2622    2621       0       0       0       0
    1187       1    2269    2273    2431    1611       0       0       0       0
    1188       1    2274    2268    2619    2272       0       0       0       0
    1189       1    2268    2274    2273    2269       0       0       0       0
    1190       1    1627    2287    1647    1632       0       0       0       0
    1191       1    2288    1627    1611    2431       0       0       0       0
    1192       1    2287    1627    2288    2288       0       0       0       0
    1193       1    2287    1923    1921    1647       0       0       0       0
    1194       1    1924    2288    2431    1919       0       0       0       0
    1195       1    2288    1924    1923    2287       0       0       0       0
    1196       1    2639    2647    2306    2300       0       0       0       0
    1197       1    2306    1682    1683    2300       0       0       0       0
    1198       1    1680    2300    1683    1684       0       0       0       0
    1199       1    2311    1682    2306    2314       0       0       0       0
    1200       1    2647    2308    2314    2306       0       0       0       0
    1201       1    2308    2332    2310    2314       0       0       0       0
    1202       1    2311    2314    2310    2312       0       0       0       0
    1203       1    2310    1731    2313    2312       0       0       0       0
    1204       1    2328    2654    2660    1736       0       0       0       0
    1205       1    2322    2328    1736    1736       0       0       0       0
    1206       1    1773    1733    1734    1734       0       0       0       0
    1207       1    2339    1773    1734    2662       0       0       0       0
    1208       1    2660    2661    1735    1736       0       0       0       0
    1209       1    2661    2662    1734    1735       0       0       0       0
    1210       1    2328    2310    2332    2654       0       0       0       0
    1211       1    1731    2310    2335    2335       0       0       0       0
    1212       1    2310    2328    2322    2335       0       0       0       0
    1213       1    2340    2350    2670    1776       0       0       0       0
    1214       1    2357    2347    2339    2663       0       0       0       0
    1215       1    2687    2346    2347    2357       0       0       0       0
    1216       1    1755    2682    2350    2340       0       0       0       0
    1217       1    1756    2352    2682    1755       0       0       0       0
    1218       1    2686    1759    2346    2687       0       0       0       0
    1219       1    1757    2353    2352    1756       0       0       0       0
    1220       1    2354    2344    1759    2686       0       0       0       0
    1221       1    2353    1757    2344    2354       0       0       0       0
    1222       1    2003    1783    2670    2669       0       0       0       0
    1223       1    2360    2004    1939    1788       0       0       0       0
    1224       1    2004    2360    1783    2003       0       0       0       0
    1225       1    2414    2371    2380    2366       0       0       0       0
    1226       1    2414    2409    2372    2372       0       0       0       0
    1227       1    2414    2372    2374    2371       0       0       0       0
    1228       1    2380    2371    2377    2697       0       0       0       0
    1229       1    2371    2374    2377    2377       0       0       0       0
    1230       1    1793    2379    1795    1795       0       0       0       0
    1231       1    2379    2696    2699    1795       0       0       0       0
    1232       1    2698    2697    2377    1796       0       0       0       0
    1233       1    2699    2698    1796    1795       0       0       0       0
    1234       1    2716    2398    1350    2706       0       0       0       0
    1235       1    1829    2709    2399    2696       0       0       0       0
    1236       1    1830    2710    2709    1829       0       0       0       0
    1237       1    2715    2397    2398    2716       0       0       0       0
    1238       1    2714    2396    2397    2715       0       0       0       0
    1239       1    1831    2711    2710    1830       0       0       0       0
    1240       1    2713    1833    2396    2714       0       0       0       0
    1241       1    1832    2712    2711    1831       0       0       0       0
    1242       1    1833    2713    2712    1832       0       0       0       0
    1243       1    2366    2415    2416    2414       0       0       0       0
    1244       1    2416    2410    2412    2414       0       0       0       0
    1245       1    2409    2414    2412    2411       0       0       0       0
    1246       1    1871    2410    2416    2422       0       0       0       0
    1247       1    2415    2726    2422    2416       0       0       0       0
    1248       1    2726    2732    2613    2422       0       0       0       0
    1249       1    1871    2422    2613    1873       0       0       0       0
    1250       1    2613    1841    1872    1873       0       0       0       0
    1251       1    1931    2274    2272    2706       0       0       0       0
    1252       1    2273    1932    1919    2431       0       0       0       0
    1253       1    1932    2273    2274    1931       0       0       0       0
    1254       1    2300    2443    2740    2639       0       0       0       0
    1255       1    2300    1680    2445    2445       0       0       0       0
    1256       1    2300    2445    1952    2443       0       0       0       0
    1257       1    2443    1952    1955    1955       0       0       0       0
    1258       1    2740    2443    1955    2452       0       0       0       0
    1259       1    1983    2459    1953    1953       0       0       0       0
    1260       1    2459    1973    2450    1953       0       0       0       0
    1261       1    2451    2452    1955    1954       0       0       0       0
    1262       1    1953    2450    2451    1954       0       0       0       0
    1263       1    2481    1982    1997    2669       0       0       0       0
    1264       1    2464    2672    2671    1973       0       0       0       0
    1265       1    2465    2673    2672    2464       0       0       0       0
    1266       1    2480    2470    1982    2481       0       0       0       0
    1267       1    2479    2469    2470    2480       0       0       0       0
    1268       1    2466    2674    2673    2465       0       0       0       0
    1269       1    2676    2468    2469    2479       0       0       0       0
    1270       1    2467    2477    2674    2466       0       0       0       0
    1271       1    2468    2676    2477    2467       0       0       0       0
    1272       1    2483    2488    2486    2486       0       0       0       0
    1273       1    2488    2501    2492    2486       0       0       0       0
    1274       1    2493    1436    2484    2484       0       0       0       0
    1275       1    2114    2493    2484    2751       0       0       0       0
    1276       1    2486    2492    2752    2011       0       0       0       0
    1277       1    2752    2751    2484    2011       0       0       0       0
    1278       1    2488    2498    2524    2501       0       0       0       0
    1279       1    2518    2498    2008    2008       0       0       0       0
    1280       1    2498    2488    2483    2008       0       0       0       0
    1281       1    2766    2768    2515    2556       0       0       0       0
    1282       1    2515    2513    2058    2556       0       0       0       0
    1283       1    2054    2556    2058    2057       0       0       0       0
    1284       1    2519    2513    2515    2523       0       0       0       0
    1285       1    2768    2516    2523    2515       0       0       0       0
    1286       1    2516    2524    2498    2523       0       0       0       0
    1287       1    2519    2523    2498    2061       0       0       0       0
    1288       1    2498    2518    2060    2061       0       0       0       0
    1289       1    2528    2538    2547    2128       0       0       0       0
    1290       1    2797    2124    2114    2537       0       0       0       0
    1291       1    2544    2534    2124    2797       0       0       0       0
    1292       1    2118    2791    2538    2528       0       0       0       0
    1293       1    2530    2792    2791    2118       0       0       0       0
    1294       1    2795    2533    2534    2544       0       0       0       0
    1295       1    2120    2541    2792    2530       0       0       0       0
    1296       1    2794    2532    2533    2795       0       0       0       0
    1297       1    2541    2120    2532    2794       0       0       0       0
    1298       1    2552    2136    2547    2551       0       0       0       0
    1299       1    2135    2218    2158    2148       0       0       0       0
    1300       1    2218    2135    2136    2552       0       0       0       0
    1301       1    2556    2574    2767    2766       0       0       0       0
    1302       1    2556    2054    2560    2560       0       0       0       0
    1303       1    2556    2560    2165    2574       0       0       0       0
    1304       1    2767    2574    2565    2800       0       0       0       0
    1305       1    2574    2165    2565    2565       0       0       0       0
    1306       1    1532    2208    2166    2166       0       0       0       0
    1307       1    2208    2799    2802    2166       0       0       0       0
    1308       1    2570    2800    2565    2167       0       0       0       0
    1309       1    2802    2570    2167    2166       0       0       0       0
    1310       1    2788    2586    2211    2551       0       0       0       0
    1311       1    2579    2781    2845    2799       0       0       0       0
    1312       1    2201    2590    2781    2579       0       0       0       0
    1313       1    2787    2206    2586    2788       0       0       0       0
    1314       1    2594    2205    2206    2787       0       0       0       0
    1315       1    2581    2783    2590    2201       0       0       0       0
    1316       1    2785    2204    2205    2594       0       0       0       0
    1317       1    2203    2784    2783    2581       0       0       0       0
    1318       1    2204    2785    2784    2203       0       0       0       0
    1319       1    2221    2603    2601    2601       0       0       0       0
    1320       1    2603    2810    2813    2601       0       0       0       0
    1321       1    2253    2243    2599    2599       0       0       0       0
    1322       1    2809    2253    2599    2811       0       0       0       0
    1323       1    2601    2813    2812    2600       0       0       0       0
    1324       1    2812    2811    2599    2600       0       0       0       0
    1325       1    2603    2613    2732    2810       0       0       0       0
    1326       1    1841    2613    2222    2222       0       0       0       0
    1327       1    2613    2603    2221    2222       0       0       0       0
    1328       1    2621    2631    2272    2619       0       0       0       0
    1329       1    2638    2628    2809    2814       0       0       0       0
    1330       1    2637    2627    2628    2638       0       0       0       0
    1331       1    2622    2632    2631    2621       0       0       0       0
    1332       1    2623    2633    2632    2622       0       0       0       0
    1333       1    2636    2626    2627    2637       0       0       0       0
    1334       1    2624    2634    2633    2623       0       0       0       0
    1335       1    2635    2625    2626    2636       0       0       0       0
    1336       1    2634    2624    2625    2635       0       0       0       0
    1337       1    2651    2639    2740    2650       0       0       0       0
    1338       1    2639    2651    2647    2647       0       0       0       0
    1339       1    2651    2649    2308    2647       0       0       0       0
    1340       1    2649    2651    2650    2648       0       0       0       0
    1341       1    2332    2649    2648    2654       0       0       0       0
    1342       1    2332    2308    2649    2649       0       0       0       0
    1343       1    2665    2660    2654    2648       0       0       0       0
    1344       1    2662    2667    2663    2339       0       0       0       0
    1345       1    2661    2666    2667    2662       0       0       0       0
    1346       1    2666    2661    2660    2665       0       0       0       0
    1347       1    2350    2481    2669    2670       0       0       0       0
    1348       1    2672    2357    2663    2671       0       0       0       0
    1349       1    2682    2480    2481    2350       0       0       0       0
    1350       1    2673    2687    2357    2672       0       0       0       0
    1351       1    2352    2479    2480    2682       0       0       0       0
    1352       1    2674    2686    2687    2673       0       0       0       0
    1353       1    2353    2676    2479    2352       0       0       0       0
    1354       1    2477    2354    2686    2674       0       0       0       0
    1355       1    2354    2477    2676    2353       0       0       0       0
    1356       1    2731    2366    2380    2730       0       0       0       0
    1357       1    2366    2731    2415    2415       0       0       0       0
    1358       1    2697    2703    2730    2380       0       0       0       0
    1359       1    2838    2699    2696    2399       0       0       0       0
    1360       1    2704    2698    2699    2838       0       0       0       0
    1361       1    2698    2704    2703    2697       0       0       0       0
    1362       1    2631    2716    2706    2272       0       0       0       0
    1363       1    2709    2638    2814    2399       0       0       0       0
    1364       1    2632    2715    2716    2631       0       0       0       0
    1365       1    2710    2637    2638    2709       0       0       0       0
    1366       1    2633    2714    2715    2632       0       0       0       0
    1367       1    2711    2636    2637    2710       0       0       0       0
    1368       1    2634    2713    2714    2633       0       0       0       0
    1369       1    2712    2635    2636    2711       0       0       0       0
    1370       1    2635    2712    2713    2634       0       0       0       0
    1371       1    2731    2729    2726    2415       0       0       0       0
    1372       1    2729    2731    2730    2815       0       0       0       0
    1373       1    2732    2729    2815    2810       0       0       0       0
    1374       1    2732    2726    2729    2729       0       0       0       0
    1375       1    2452    2826    2650    2740       0       0       0       0
    1376       1    2828    2450    1973    2671       0       0       0       0
    1377       1    2451    2827    2826    2452       0       0       0       0
    1378       1    2827    2451    2450    2828       0       0       0       0
    1379       1    2841    2492    2501    2774       0       0       0       0
    1380       1    2751    2758    2537    2114       0       0       0       0
    1381       1    2842    2752    2492    2841       0       0       0       0
    1382       1    2752    2842    2758    2751       0       0       0       0
    1383       1    2524    2775    2774    2501       0       0       0       0
    1384       1    2524    2516    2775    2775       0       0       0       0
    1385       1    2777    2766    2767    2844       0       0       0       0
    1386       1    2766    2777    2768    2768       0       0       0       0
    1387       1    2777    2775    2516    2768       0       0       0       0
    1388       1    2775    2777    2844    2774       0       0       0       0
    1389       1    2538    2788    2551    2547       0       0       0       0
    1390       1    2781    2797    2537    2845       0       0       0       0
    1391       1    2791    2787    2788    2538       0       0       0       0
    1392       1    2590    2544    2797    2781       0       0       0       0
    1393       1    2792    2594    2787    2791       0       0       0       0
    1394       1    2783    2795    2544    2590       0       0       0       0
    1395       1    2541    2785    2594    2792       0       0       0       0
    1396       1    2784    2794    2795    2783       0       0       0       0
    1397       1    2794    2784    2785    2541       0       0       0       0
    1398       1    2800    2806    2844    2767       0       0       0       0
    1399       1    2808    2802    2799    2845       0       0       0       0
    1400       1    2807    2570    2802    2808       0       0       0       0
    1401       1    2570    2807    2806    2800       0       0       0       0
    1402       1    2816    2813    2810    2815       0       0       0       0
    1403       1    2811    2818    2814    2809       0       0       0       0
    1404       1    2817    2812    2813    2816       0       0       0       0
    1405       1    2812    2817    2818    2811       0       0       0       0
    1406       1    2648    2650    2826    2665       0       0       0       0
    1407       1    2671    2663    2667    2828       0       0       0       0
    1408       1    2665    2826    2827    2666       0       0       0       0
    1409       1    2828    2667    2666    2827       0       0       0       0
    1410       1    2815    2730    2703    2816       0       0       0       0
    1411       1    2399    2814    2818    2838       0       0       0       0
    1412       1    2816    2703    2704    2817       0       0       0       0
    1413       1    2838    2818    2817    2704       0       0       0       0
    1414       1    2774    2844    2806    2841       0       0       0       0
    1415       1    2845    2537    2758    2808       0       0       0       0
    1416       1    2841    2806    2807    2842       0       0       0       0
    1417       1    2808    2758    2842    2807       0       0       0       0
*NODE
$#   nid               x               y               z      tc      rc  
       1    5.000000e-04        0.029628       0.0211739       0       0
       3   -5.000000e-04        0.038802        0.005884       0       0
       4   -5.000000e-04       0.0306473       0.0194751       0       0
       7   -5.000000e-04       0.0337053       0.0143784       0       0
      10   -5.000000e-04       0.0367633       0.0092818       0       0
      11   -5.000000e-04       0.0377826       0.0075829       0       0
      13    5.000000e-04       0.0377826       0.0075829       0       0
      15    5.000000e-04        0.035744       0.0109807       0       0
      17    5.000000e-04       0.0337053       0.0143784       0       0
      19    5.000000e-04       0.0316667       0.0177762       0       0
      20    5.000000e-04       0.0306473       0.0194751       0       0
      24    5.000000e-04       0.0274712       0.0238234       0       0
      25   -5.000000e-04       0.0261324       0.0248913       0       0
      27   -5.000000e-04       0.0274712       0.0238234       0       0
      28   -5.000000e-04       0.0286444       0.0225758       0       0
      31         -0.0021       0.0299613       0.0190635       0       0
      34         -0.0021       0.0330193       0.0139668       0       0
      36         -0.0021        0.035058       0.0105691       0       0
      38         -0.0021       0.0370966       0.0071713       0       0
      40   -5.000000e-04        0.029628        0.021174       0       0
      42   -5.000000e-04       0.0316667       0.0177762       0       0
      43   -5.000000e-04        0.032686       0.0160773       0       0
      45   -5.000000e-04       0.0347246       0.0126796       0       0
      46   -5.000000e-04        0.035744       0.0109807       0       0
      50    5.000000e-04        0.038802        0.005884       0       0
      52    5.000000e-04       0.0409028       0.0035695       0       0
      53    5.000000e-04       0.0397382       0.0046232       0       0
      58   -5.000000e-04       0.0397382       0.0046232       0       0
      62          0.0021       0.0360773       0.0088702       0       0
      63          0.0021        0.035058       0.0105691       0       0
      68          0.0021       0.0299613       0.0190635       0       0
      72    5.000000e-04       0.0367633       0.0092818       0       0
      74    5.000000e-04       0.0347246       0.0126796       0       0
      76    5.000000e-04        0.032686       0.0160773       0       0
      80         -0.0021       0.0256794       0.0242319       0       0
      81         -0.0021        0.028024       0.0220708       0       0
      90    5.000000e-04       0.0236499       0.0261985       0       0
      91    5.000000e-04       0.0249295       0.0256177       0       0
      94   -5.000000e-04       0.0236499       0.0261985       0       0
      98          0.0021        0.028024       0.0220708       0       0
      99          0.0021        0.026929       0.0232352       0       0
     101    5.000063e-04       0.0261324       0.0248913       0       0
     103    5.000000e-04       0.0286444       0.0225758       0       0
     106         -0.0021        0.038116       0.0054724       0       0
     109         -0.0021       0.0391459       0.0040855       0       0
     113   -5.000000e-04       0.0422508       0.0027637       0       0
     114   -5.000000e-04       0.0409028       0.0035695       0       0
     116         -0.0045        0.027913        0.020145       0       0
     117         -0.0045        0.037087        0.004855       0       0
     118         -0.0045       0.0289324       0.0184461       0       0
     121         -0.0045       0.0319903       0.0133494       0       0
     126         -0.0021        0.028942       0.0207624       0       0
     130         -0.0021       0.0309807       0.0173646       0       0
     131         -0.0021           0.032       0.0156657       0       0
     133         -0.0021       0.0340387        0.012268       0       0
     135         -0.0021       0.0360773       0.0088702       0       0
     142    5.000000e-04       0.0437302       0.0022367       0       0
     144   -5.000000e-04       0.0437302       0.0022367       0       0
     146          0.0021        0.038116       0.0054724       0       0
     148          0.0021       0.0391459       0.0040855       0       0
     153    5.000000e-04       0.0422508       0.0027637       0       0
     156          0.0045        0.037087        0.004855       0       0
     160          0.0045        0.034029       0.0099517       0       0
     161          0.0045       0.0330097       0.0116506       0       0
     162          0.0045       0.0319903       0.0133494       0       0
     164          0.0045       0.0299517       0.0167472       0       0
     165          0.0045       0.0289324       0.0184461       0       0
     169          0.0021       0.0370966       0.0071713       0       0
     172          0.0021       0.0340387        0.012268       0       0
     173          0.0021       0.0330193       0.0139668       0       0
     174          0.0021           0.032       0.0156657       0       0
     175          0.0021       0.0309807       0.0173646       0       0
     182         -0.0021       0.0239675       0.0251984       0       0
     185   -5.000000e-04       0.0223112       0.0266258       0       0
     188   -5.000000e-04       0.0249295       0.0256177       0       0
     191         -0.0045       0.0261157       0.0223528       0       0
     192         -0.0045       0.0270933       0.0213132       0       0
     198         -0.0021        0.026929       0.0232352       0       0
     199      -0.0031584       0.0281846       0.0203079       0       0
     200       -0.003223        0.026831        0.022046       0       0
     201    5.000001e-04       0.0190636       0.0258007       0       0
     207          0.0021        0.022113       0.0258508       0       0
     209          0.0021       0.0239675       0.0251984       0       0
     211    5.000000e-04       0.0223112       0.0266258       0       0
     220          0.0045       0.0261157       0.0223529       0       0
     221          0.0021        0.028942       0.0207624       0       0
     222       0.0031584       0.0281846       0.0203079       0       0
     228       0.0031584       0.0251793       0.0235038       0       0
     229        0.003223        0.026831        0.022046       0       0
     235   -5.000000e-04       0.0447446       0.0018055       0       0
     237         -0.0045       0.0432473    2.959063e-04       0       0
     239         -0.0045        0.039713       0.0019619       0       0
     240         -0.0045        0.041398    9.546103e-04       0       0
     242      -0.0031584       0.0373586        0.005018       0       0
     244         -0.0021       0.0419097       0.0020401       0       0
     245         -0.0021       0.0404269       0.0029265       0       0
     253          -0.008        0.027913        0.020145       0       0
     254        -0.00625        0.027913        0.020145       0       0
     255          -0.008        0.037087        0.004855       0       0
     256          -0.008       0.0289324       0.0184461       0       0
     257          -0.008       0.0299517       0.0167472       0       0
     258          -0.008        0.030971       0.0150483       0       0
     259          -0.008       0.0319903       0.0133494       0       0
     260          -0.008       0.0330097       0.0116506       0       0
     261          -0.008        0.034029       0.0099517       0       0
     262          -0.008       0.0350483       0.0082528       0       0
     263          -0.008       0.0360676       0.0065539       0       0
     265        -0.00625        0.037087        0.004855       0       0
     267         -0.0045       0.0299517       0.0167472       0       0
     268         -0.0045        0.030971       0.0150483       0       0
     270         -0.0045       0.0330097       0.0116506       0       0
     271         -0.0045        0.034029       0.0099517       0       0
     272         -0.0045       0.0350483       0.0082528       0       0
     273         -0.0045       0.0360676       0.0065539       0       0
     274      -0.0062501       0.0289324       0.0184461       0       0
     275      -0.0062504       0.0360676       0.0065539       0       0
     276      -0.0062528       0.0350483       0.0082528       0       0
     277      -0.0062507       0.0299517       0.0167472       0       0
     278      -0.0062508        0.030971       0.0150483       0       0
     279      -0.0062516       0.0319903       0.0133494       0       0
     280      -0.0062513        0.034029       0.0099517       0       0
     281      -0.0062504       0.0330097       0.0116506       0       0
     283    5.000000e-04       0.0455745       0.0010801       0       0
     288    5.000000e-04       0.0447446       0.0018055       0       0
     289   -5.000000e-04       0.0478798   -2.328306e-10       0       0
     294    5.000000e-04       0.0466069    2.834215e-04       0       0
     297          0.0045       0.0382573        0.003279       0       0
     299          0.0045        0.041398    9.546103e-04       0       0
     301       0.0031584       0.0433238    6.032456e-04       0       0
     304          0.0021       0.0404269       0.0029265       0       0
     305          0.0021       0.0419097       0.0020401       0       0
     306       0.0031584       0.0373586        0.005018       0       0
     311         0.00625        0.037087        0.004855       0       0
     312           0.008        0.027913        0.020145       0       0
     313           0.008       0.0360676       0.0065539       0       0
     314           0.008       0.0350483       0.0082528       0       0
     315           0.008        0.034029       0.0099517       0       0
     316           0.008       0.0330097       0.0116506       0       0
     317           0.008       0.0319903       0.0133494       0       0
     318           0.008        0.030971       0.0150483       0       0
     319           0.008       0.0299517       0.0167472       0       0
     320           0.008       0.0289324       0.0184461       0       0
     321          0.0045        0.027913        0.020145       0       0
     322         0.00625        0.027913        0.020145       0       0
     323          0.0045       0.0360676       0.0065539       0       0
     324          0.0045       0.0350483       0.0082528       0       0
     328          0.0045        0.030971       0.0150483       0       0
     331       0.0062496       0.0360676       0.0065539       0       0
     332       0.0062499       0.0289324       0.0184461       0       0
     333       0.0062472       0.0350483       0.0082528       0       0
     334       0.0062493       0.0299517       0.0167472       0       0
     335       0.0062492        0.030971       0.0150483       0       0
     336       0.0062484       0.0319903       0.0133494       0       0
     337       0.0062487        0.034029       0.0099517       0       0
     338       0.0062496       0.0330097       0.0116506       0       0
     340   -5.000000e-04       0.0190636       0.0258007       0       0
     343   -5.000000e-04       0.0205822       0.0266272       0       0
     346      -0.0031584       0.0251793       0.0235038       0       0
     356      -0.0032091       0.0235922       0.0243716       0       0
     357         -0.0045           0.025       0.0232428       0       0
     361         -0.0045       0.0218157       0.0246882       0       0
     363         -0.0045       0.0234715       0.0241058       0       0
     366      -0.0174861       0.0193381           0.025       0       0
     367         -0.0058       0.0193381           0.025       0       0
     369      -0.0096954       0.0193381           0.025       0       0
     370      -0.0116431       0.0193381           0.025       0       0
     372      -0.0155385       0.0193381           0.025       0       0
     373          -0.015            0.02       0.0249781       0       0
     374      -0.0162864       0.0198317       0.0249878       0       0
     375          -0.013            0.02       0.0249781       0       0
     376          -0.008           0.025       0.0232427       0       0
     377      -0.0082014       0.0235958       0.0240481       0       0
     378      -0.0088562        0.022202       0.0245811       0       0
     379       -0.009957       0.0210325       0.0248557       0       0
     380      -0.0113917       0.0202656       0.0249572       0       0
     382          -0.008       0.0261157       0.0223528       0       0
     383          -0.008       0.0270933       0.0213132       0       0
     385      -0.0058219       0.0204079       0.0249426       0       0
     386      -0.0067136       0.0219254       0.0246595       0       0
     387      -0.0062503       0.0270918       0.0213151       0       0
     388      -0.0062562       0.0261061       0.0223617       0       0
     389      -0.0063005        0.024918       0.0232985       0       0
     390      -0.0064463       0.0235334       0.0240774       0       0
     391      -0.0077989       0.0205178       0.0249302       0       0
     392       0.0018409       0.0202379       0.0259642       0       0
     398    5.000000e-04       0.0205822       0.0266272       0       0
     402    5.000000e-04       0.0181191       0.0252074       0       0
     404   -5.000000e-04       0.0181191       0.0252074       0       0
     407       0.0031583       0.0218942       0.0249951       0       0
     410          0.0045       0.0234715       0.0241058       0       0
     411          0.0021       0.0256794       0.0242319       0       0
     415       0.0032091       0.0235922       0.0243716       0       0
     416          0.0045       0.0218157       0.0246882       0       0
     418       0.0043409       0.0205439        0.024927       0       0
     419          0.0045           0.025       0.0232428       0       0
     424          0.0045       0.0270933       0.0213132       0       0
     428           0.008           0.025       0.0232427       0       0
     429           0.008       0.0261157       0.0223528       0       0
     430           0.008       0.0270933       0.0213132       0       0
     431           0.013            0.02       0.0249781       0       0
     432       0.0113917       0.0202656       0.0249572       0       0
     433        0.009957       0.0210325       0.0248557       0       0
     434       0.0088562        0.022202       0.0245811       0       0
     435       0.0082014       0.0235958       0.0240481       0       0
     436           0.015            0.02       0.0249781       0       0
     438       0.0162864       0.0198317       0.0249878       0       0
     440       0.0135908       0.0193381           0.025       0       0
     441       0.0116431       0.0193381           0.025       0       0
     442       0.0096954       0.0193381           0.025       0       0
     443       0.0077477       0.0193381           0.025       0       0
     444          0.0058       0.0193381           0.025       0       0
     445       0.0058219       0.0204079       0.0249426       0       0
     446       0.0067136       0.0219254       0.0246595       0       0
     447       0.0062503       0.0270918       0.0213151       0       0
     448       0.0062562        0.026106       0.0223617       0       0
     449       0.0063005        0.024918       0.0232985       0       0
     450       0.0064463       0.0235334       0.0240774       0       0
     451       0.0077989       0.0205178       0.0249302       0       0
     452      -0.0040693       0.0456619   -5.820766e-11       0       0
     454      -0.0031724       0.0468199   -5.820766e-11       0       0
     456      -0.0019277       0.0444955       0.0012527       0       0
     457      -0.0028655       0.0450063    3.314966e-04       0       0
     458   -5.000000e-04       0.0455745       0.0010801       0       0
     459   -5.000000e-04       0.0466069    2.834207e-04       0       0
     463         -0.0021        0.043537       0.0014604       0       0
     465      -0.0031583       0.0433238    6.032533e-04       0       0
     469         -0.0045       0.0382573        0.003279       0       0
     474       -0.004395       0.0444651    7.187203e-05       0       0
     476      -0.0060347       0.0456619   -9.313226e-10       0       0
     478          -0.008       0.0380923       0.0034654       0       0
     479          -0.008       0.0393203        0.002268       0       0
     480          -0.008       0.0407349        0.001298       0       0
     481          -0.008       0.0422944    5.840436e-04       0       0
     482          -0.008        0.043953    1.470922e-04       0       0
     485      -0.0064102        0.043123    3.276709e-04       0       0
     486      -0.0062501       0.0381743       0.0033716       0       0
     487      -0.0062525        0.039519       0.0021092       0       0
     488      -0.0062701       0.0411038       0.0010992       0       0
     492       0.0028655       0.0450063    3.314935e-04       0       0
     494        0.004395       0.0444651    7.187109e-05       0       0
     495          0.0021        0.043537       0.0014604       0       0
     500       0.0031724       0.0468199             0.0       0       0
     504       0.0019277       0.0444955       0.0012527       0       0
     506           0.008       0.0456619             0.0       0       0
     507       0.0040693       0.0456619   -6.146209e-10       0       0
     509           0.008           0.057             0.0       0       0
     510           0.008       0.0551103             0.0       0       0
     511           0.008       0.0532206             0.0       0       0
     512           0.008        0.051331             0.0       0       0
     513           0.008       0.0494413             0.0       0       0
     514           0.008       0.0475516             0.0       0       0
     515           0.005            0.06             0.0       0       0
     516          0.0065       0.0595981             0.0       0       0
     517       0.0075981          0.0585             0.0       0       0
     518          -0.005            0.06             0.0       0       0
     519          -0.003            0.06             0.0       0       0
     520          -0.001            0.06             0.0       0       0
     521           0.001            0.06             0.0       0       0
     522           0.003            0.06             0.0       0       0
     523          -0.008           0.057             0.0       0       0
     524      -0.0075981          0.0585             0.0       0       0
     525         -0.0065       0.0595981             0.0       0       0
     526          -0.008       0.0456619             0.0       0       0
     527          -0.008       0.0551103             0.0       0       0
     528          -0.008       0.0532206             0.0       0       0
     529          -0.008        0.051331             0.0       0       0
     530          -0.008       0.0494413             0.0       0       0
     531          -0.008       0.0475516             0.0       0       0
     536      -0.0019359       0.0476024             0.0       0       0
     537    4.999999e-04       0.0478798    2.785550e-16       0       0
     539       0.0019359       0.0476024             0.0       0       0
     540             0.0           0.056             0.0       0       0
     541    4.592426e-20            0.05             0.0       0       0
     542      -0.0017634       0.0554271             0.0       0       0
     543      -0.0028532       0.0539271             0.0       0       0
     544      -0.0028532       0.0520729             0.0       0       0
     545      -0.0017634        0.050573             0.0       0       0
     546       0.0017634        0.050573             0.0       0       0
     547       0.0028532       0.0520729             0.0       0       0
     548       0.0028532       0.0539271             0.0       0       0
     549       0.0017634       0.0554271             0.0       0       0
     550       0.0058946       0.0472682             0.0       0       0
     551      -0.0058995       0.0473859             0.0       0       0
     552       0.0057038        0.048881             0.0       0       0
     554      -0.0060553       0.0490107             0.0       0       0
     556       0.0059701       0.0532778             0.0       0       0
     557      -0.0058767       0.0532916             0.0       0       0
     558       0.0061826       0.0549954             0.0       0       0
     559      -0.0060091       0.0551529             0.0       0       0
     560       0.0057868       0.0571514             0.0       0       0
     561      -0.0060852       0.0571712             0.0       0       0
     562       0.0041818       0.0484577             0.0       0       0
     563       0.0031646       0.0491636             0.0       0       0
     564      -0.0038749       0.0492547             0.0       0       0
     566      -0.0054399       0.0512908             0.0       0       0
     567      -0.0045342       0.0534096             0.0       0       0
     568       -0.004172       0.0547173             0.0       0       0
     573      -0.0034014        0.057286             0.0       0       0
     574      -0.0010321       0.0581235             0.0       0       0
     575    9.468379e-04       0.0581464             0.0       0       0
     576       0.0029076       0.0574702             0.0       0       0
     577       0.0045615        0.058324             0.0       0       0
     578       0.0052236       0.0510105             0.0       0       0
     579       0.0045995       0.0536858             0.0       0       0
     580       0.0049311       0.0547932             0.0       0       0
     581       0.0041943       0.0557221             0.0       0       0
     583          0.0045       0.0432473    2.959035e-04       0       0
     587          0.0045        0.039713       0.0019619       0       0
     589           0.008        0.037087        0.004855       0       0
     592           0.008        0.043953    1.470922e-04       0       0
     593           0.008       0.0422944    5.840436e-04       0       0
     594           0.008       0.0407349        0.001298       0       0
     595           0.008       0.0393203        0.002268       0       0
     596           0.008       0.0380923       0.0034654       0       0
     597       0.0060347       0.0456619   -9.313226e-10       0       0
     600       0.0062499       0.0381739        0.003372       0       0
     601       0.0062524       0.0395188       0.0021094       0       0
     602       0.0062697       0.0411025       0.0010999       0       0
     603        0.006411       0.0431248    3.271997e-04       0       0
     606      -0.0043409       0.0205439        0.024927       0       0
     607      -0.0018409       0.0202379       0.0259642       0       0
     608      -0.0027546       0.0198292       0.0252516       0       0
     609         -0.0021        0.022113       0.0258508       0       0
     610      -0.0031583       0.0218942       0.0249951       0       0
     611      -0.0038524       0.0193381           0.025       0       0
     613      -0.0030606       0.0181469           0.025       0       0
     624       0.0193381       0.0096954           0.025       0       0
     627       0.0174861       0.0193381           0.025       0       0
     628       0.0185355       0.0185355           0.025       0       0
     630       0.0155385       0.0193381           0.025       0       0
     637       0.0018955       0.0173208           0.025       0       0
     639   -4.999998e-04       0.0170232           0.025       0       0
     642      -0.0018955       0.0173208           0.025       0       0
     645      -0.0077477       0.0193381           0.025       0       0
     648      -0.0135908       0.0193381           0.025       0       0
     650      -0.0193381       0.0174861           0.025       0       0
     651      -0.0185355       0.0185355           0.025       0       0
     655      -0.0193381       0.0096954           0.025       0       0
     660      -0.0173208       0.0018955           0.025       0       0
     667      -0.0193381      -0.0155385           0.025       0       0
     668      -0.0193381      -0.0135908           0.025       0       0
     669      -0.0193381      -0.0116431           0.025       0       0
     671      -0.0193381      -0.0077477           0.025       0       0
     672      -0.0193381         -0.0058           0.025       0       0
     673      -0.0174861      -0.0193381           0.025       0       0
     674      -0.0185355      -0.0185355           0.025       0       0
     676         -0.0058      -0.0193381           0.025       0       0
     677      -0.0077477      -0.0193381           0.025       0       0
     678      -0.0096954      -0.0193381           0.025       0       0
     679      -0.0116431      -0.0193381           0.025       0       0
     680      -0.0135908      -0.0193381           0.025       0       0
     681      -0.0155385      -0.0193381           0.025       0       0
     682   -5.000002e-04      -0.0170232           0.025       0       0
     684      -0.0030606      -0.0181469           0.025       0       0
     687       0.0030606      -0.0181469           0.025       0       0
     689       0.0174861      -0.0193381           0.025       0       0
     693       0.0096954      -0.0193381           0.025       0       0
     696       0.0193381      -0.0174861           0.025       0       0
     697       0.0185355      -0.0185355           0.025       0       0
     699       0.0193381         -0.0058           0.025       0       0
     700       0.0193381      -0.0077477           0.025       0       0
     701       0.0193381      -0.0096954           0.025       0       0
     702       0.0193381      -0.0116431           0.025       0       0
     703       0.0193381      -0.0135908           0.025       0       0
     705       0.0170232   -5.000002e-04           0.025       0       0
     706       0.0173208      -0.0018955           0.025       0       0
     707       0.0181469      -0.0030606           0.025       0       0
     708       0.0170232    4.999998e-04           0.025       0       0
     711          0.0125             0.0           0.025       0       0
     712         -0.0125   -1.913511e-19           0.025       0       0
     713       0.0123461       0.0019554           0.025       0       0
     714       0.0118882       0.0038627           0.025       0       0
     715       0.0111376       0.0056749           0.025       0       0
     716       0.0101127       0.0073473           0.025       0       0
     717       0.0088388       0.0088388           0.025       0       0
     718       0.0073473       0.0101127           0.025       0       0
     719       0.0056749       0.0111376           0.025       0       0
     720       0.0038627       0.0118882           0.025       0       0
     721       0.0019554       0.0123461           0.025       0       0
     722    7.654043e-19          0.0125           0.025       0       0
     723      -0.0019554       0.0123461           0.025       0       0
     724      -0.0038627       0.0118882           0.025       0       0
     725      -0.0056749       0.0111376           0.025       0       0
     726      -0.0073473       0.0101127           0.025       0       0
     727      -0.0088388       0.0088388           0.025       0       0
     728      -0.0101127       0.0073473           0.025       0       0
     729      -0.0111376       0.0056749           0.025       0       0
     730      -0.0118882       0.0038627           0.025       0       0
     731      -0.0123461       0.0019554           0.025       0       0
     732      -0.0123461      -0.0019554           0.025       0       0
     733      -0.0118882      -0.0038627           0.025       0       0
     734      -0.0111376      -0.0056749           0.025       0       0
     735      -0.0101127      -0.0073473           0.025       0       0
     736      -0.0088388      -0.0088388           0.025       0       0
     737      -0.0073473      -0.0101127           0.025       0       0
     738      -0.0056749      -0.0111376           0.025       0       0
     739      -0.0038627      -0.0118882           0.025       0       0
     740      -0.0019554      -0.0123461           0.025       0       0
     741   -7.654043e-19         -0.0125           0.025       0       0
     742       0.0019554      -0.0123461           0.025       0       0
     743       0.0038627      -0.0118882           0.025       0       0
     744       0.0056749      -0.0111376           0.025       0       0
     745       0.0073473      -0.0101127           0.025       0       0
     746       0.0088388      -0.0088388           0.025       0       0
     747       0.0101127      -0.0073473           0.025       0       0
     748       0.0111376      -0.0056749           0.025       0       0
     749       0.0118882      -0.0038627           0.025       0       0
     750       0.0123461      -0.0019554           0.025       0       0
     751       0.0155783        0.012956           0.025       0       0
     752       0.0105846       0.0132069           0.025       0       0
     753       0.0149357       0.0147583           0.025       0       0
     754       0.0132069       0.0155783           0.025       0       0
     755       0.0114046       0.0149357           0.025       0       0
     756       0.0112272       0.0114046           0.025       0       0
     757        0.012956       0.0105846           0.025       0       0
     758       0.0147583       0.0112272           0.025       0       0
     759       0.0105846      -0.0132069           0.025       0       0
     760       0.0155783       -0.012956           0.025       0       0
     761       0.0114046      -0.0149357           0.025       0       0
     762       0.0132069      -0.0155783           0.025       0       0
     763       0.0149357      -0.0147583           0.025       0       0
     764       0.0147583      -0.0112272           0.025       0       0
     765        0.012956      -0.0105846           0.025       0       0
     766       0.0112272      -0.0114046           0.025       0       0
     767      -0.0105846       0.0132069           0.025       0       0
     768      -0.0155783        0.012956           0.025       0       0
     769      -0.0114046       0.0149357           0.025       0       0
     770      -0.0132069       0.0155783           0.025       0       0
     771      -0.0149357       0.0147583           0.025       0       0
     772      -0.0147583       0.0112272           0.025       0       0
     773       -0.012956       0.0105846           0.025       0       0
     774      -0.0112272       0.0114046           0.025       0       0
     775      -0.0155783       -0.012956           0.025       0       0
     776      -0.0105846      -0.0132069           0.025       0       0
     777      -0.0149357      -0.0147583           0.025       0       0
     778      -0.0132069      -0.0155783           0.025       0       0
     779      -0.0114046      -0.0149357           0.025       0       0
     780      -0.0112272      -0.0114046           0.025       0       0
     781       -0.012956      -0.0105846           0.025       0       0
     782      -0.0147583      -0.0112272           0.025       0       0
     785    1.016388e-05      -0.0151005           0.025       0       0
     787       0.0028385      -0.0149875           0.025       0       0
     791        0.015101    3.235963e-06           0.025       0       0
     793       0.0150042       0.0027094           0.025       0       0
     796       0.0027578       0.0150417           0.025       0       0
     797    5.980254e-06       0.0151074           0.025       0       0
     799      -0.0026978       0.0150335           0.025       0       0
     804      -0.0150778   -6.316265e-06           0.025       0       0
     805      -0.0150338       0.0026637           0.025       0       0
     807       0.0078606       0.0175294           0.025       0       0
     808       0.0052245       0.0170218           0.025       0       0
     809       0.0093384       0.0179354           0.025       0       0
     810       0.0108822       0.0169515           0.025       0       0
     811       0.0133149       0.0171885           0.025       0       0
     812       0.0153293       0.0172558           0.025       0       0
     813       0.0172298       0.0171947           0.025       0       0
     816       0.0176231       0.0136577           0.025       0       0
     817       0.0175088       0.0118125           0.025       0       0
     819       0.0170984       0.0075332           0.025       0       0
     820       0.0169848       0.0050709           0.025       0       0
     821      -0.0166945      -0.0052099           0.025       0       0
     822      -0.0170929      -0.0076717           0.025       0       0
     824      -0.0175336      -0.0118563           0.025       0       0
     825      -0.0176478       -0.013653           0.025       0       0
     826      -0.0174902      -0.0152247           0.025       0       0
     827      -0.0176269      -0.0164794           0.025       0       0
     829      -0.0161236      -0.0173329           0.025       0       0
     830      -0.0134836      -0.0172338           0.025       0       0
     831      -0.0114152      -0.0172735           0.025       0       0
     832      -0.0094513      -0.0173155           0.025       0       0
     833      -0.0075151      -0.0173364           0.025       0       0
     834      -0.0050731      -0.0170742           0.025       0       0
     835       0.0171944      -0.0074735           0.025       0       0
     836        0.016989      -0.0050306           0.025       0       0
     837       0.0171729       -0.009443           0.025       0       0
     838       0.0172236      -0.0113779           0.025       0       0
     839       0.0172258      -0.0132934           0.025       0       0
     840       0.0172902      -0.0152584           0.025       0       0
     841        0.017225      -0.0171885           0.025       0       0
     843       0.0153337      -0.0172539           0.025       0       0
     844       0.0133897      -0.0172351           0.025       0       0
     845       0.0114682       -0.017344           0.025       0       0
     846       0.0095934      -0.0174033           0.025       0       0
     847       0.0077291      -0.0173764           0.025       0       0
     848       0.0052595      -0.0170335           0.025       0       0
     849      -0.0172259       0.0171899           0.025       0       0
     850      -0.0172895       0.0152593           0.025       0       0
     851      -0.0153337       0.0172542           0.025       0       0
     852      -0.0172242       0.0132933           0.025       0       0
     853      -0.0133907       0.0172282           0.025       0       0
     854      -0.0172123       0.0113577           0.025       0       0
     855      -0.0171527       0.0094014           0.025       0       0
     856      -0.0171745       0.0074127           0.025       0       0
     857       -0.016983       0.0049878           0.025       0       0
     858        -0.01146       0.0172634           0.025       0       0
     859      -0.0094995        0.017222           0.025       0       0
     860      -0.0075093       0.0172192           0.025       0       0
     861      -0.0050508       0.0169953           0.025       0       0
     862       0.0173367       0.0153071           0.025       0       0
     863       0.0166018       0.0099115           0.025       0       0
     864       0.0145581       0.0085529           0.025       0       0
     865       0.0129156       -0.009152           0.025       0       0
     866       0.0150388       -0.009174           0.025       0       0
     867       0.0111425      -0.0093513           0.025       0       0
     869      -0.0122913       0.0087418           0.025       0       0
     870      -0.0149631       0.0090394           0.025       0       0
     871      -0.0107568       0.0097068           0.025       0       0
     872      -0.0167458      -0.0102169           0.025       0       0
     873       -0.014756      -0.0085718           0.025       0       0
     874       -0.010171      -0.0102182           0.025       0       0
     875      -0.0087136      -0.0116422           0.025       0       0
     876      -0.0115808      -0.0086373           0.025       0       0
     877       0.0101448       0.0102253           0.025       0       0
     878       0.0086154        0.011601           0.025       0       0
     879       0.0114881       0.0087014           0.025       0       0
     880       0.0085609      -0.0134143           0.025       0       0
     881       0.0090169      -0.0112077           0.025       0       0
     882       0.0092314      -0.0155822           0.025       0       0
     883        0.008191          0.0151           0.025       0       0
     884      -0.0092489       0.0151291           0.025       0       0
     885      -0.0092197       0.0131554           0.025       0       0
     886      -0.0070193      -0.0131081           0.025       0       0
     887      -0.0046376      -0.0137021           0.025       0       0
     889       -0.013385      -0.0043139           0.025       0       0
     890      -0.0027137      -0.0150397           0.025       0       0
     891      -0.0147838      -0.0027258           0.025       0       0
     894      -0.0091108       0.0113127           0.025       0       0
     896      -0.0138436       0.0047887           0.025       0       0
     897      -0.0130659       0.0065125           0.025       0       0
     898      -0.0078715       0.0112324           0.025       0       0
     899      -0.0070278       0.0128182           0.025       0       0
     900      -0.0049124       0.0138259           0.025       0       0
     903       0.0051943         -0.0135           0.025       0       0
     905       0.0067511      -0.0121778           0.025       0       0
     906       0.0076983      -0.0111482           0.025       0       0
     907        0.012935      -0.0069281           0.025       0       0
     908       0.0138345      -0.0048815           0.025       0       0
     910       0.0150335       -0.002686           0.025       0       0
     913       0.0050742       0.0138153           0.025       0       0
     914         0.00692       0.0127601           0.025       0       0
     915       0.0129342       0.0069815           0.025       0       0
     916       0.0136288       0.0046249           0.025       0       0
     919      -0.0150847       0.0070583           0.025       0       0
     921       0.0155459       0.0073619           0.025       0       0
     922      -0.0072771       0.0151938           0.025       0       0
     923      -0.0134398      -0.0064599           0.025       0       0
     924       0.0151066       0.0061195           0.025       0       0
     925      -0.0088457      -0.0151106           0.025       0       0
     926       0.0151587      -0.0072227           0.025       0       0
     927       0.0151908      -0.0057037           0.025       0       0
     928      -0.0057462       0.0151817           0.025       0       0
     929      -0.0151979       0.0055713           0.025       0       0
     930       0.0060053       0.0152119           0.025       0       0
     932       0.0066722      -0.0150063           0.025       0       0
     933       0.0078379      -0.0159704           0.025       0       0
     934      -0.0073447      -0.0158769           0.025       0       0
     935      -0.0061641      -0.0153205           0.025       0       0
     936    5.000003e-04       0.0170232           0.025       0       0
     937       0.0038524       0.0193381           0.025       0       0
     939       0.0030606       0.0181469           0.025       0       0
     950       0.0027546       0.0198292       0.0252516       0       0
     951      -0.0218157          0.0045       0.0246882       0       0
     952      -0.0193381       0.0038524           0.025       0       0
     953      -0.0205439       0.0043409        0.024927       0       0
     954          -0.025          0.0045       0.0232427       0       0
     956      -0.0234715          0.0045       0.0241058       0       0
     961       -0.027913           0.008        0.020145       0       0
     962       -0.027913         0.00625        0.020145       0       0
     963          -0.025           0.008       0.0232427       0       0
     964      -0.0261157           0.008       0.0223528       0       0
     965      -0.0270933           0.008       0.0213132       0       0
     966           -0.02           0.013       0.0249781       0       0
     967      -0.0202656       0.0113917       0.0249572       0       0
     968      -0.0210325        0.009957       0.0248557       0       0
     969       -0.022202       0.0088562       0.0245811       0       0
     970      -0.0235958       0.0082014       0.0240481       0       0
     971           -0.02           0.015       0.0249781       0       0
     973      -0.0198317       0.0162864       0.0249878       0       0
     974      -0.0193381          0.0058           0.025       0       0
     975      -0.0193381       0.0077477           0.025       0       0
     977      -0.0193381       0.0116431           0.025       0       0
     978      -0.0193381       0.0135908           0.025       0       0
     979      -0.0193381       0.0155385           0.025       0       0
     980      -0.0204079       0.0058219       0.0249426       0       0
     981      -0.0219254       0.0067136       0.0246595       0       0
     982      -0.0270918       0.0062503       0.0213151       0       0
     983      -0.0261061       0.0062562       0.0223617       0       0
     984       -0.024918       0.0063005       0.0232985       0       0
     985      -0.0235334       0.0064463       0.0240774       0       0
     986      -0.0205178       0.0077989       0.0249302       0       0
     987           0.025          0.0045       0.0232428       0       0
     988        0.027913          0.0045        0.020145       0       0
     989       0.0261157          0.0045       0.0223528       0       0
     990       0.0270933          0.0045       0.0213132       0       0
     991       0.0218157          0.0045       0.0246882       0       0
     993       0.0234715          0.0045       0.0241058       0       0
     994       0.0193381       0.0038524           0.025       0       0
     995       0.0205439       0.0043409        0.024927       0       0
     996       0.0193381       0.0174861           0.025       0       0
     997       0.0193381       0.0155385           0.025       0       0
     998       0.0193381       0.0135908           0.025       0       0
     999       0.0193381       0.0116431           0.025       0       0
    1001       0.0193381       0.0077477           0.025       0       0
    1002       0.0193381          0.0058           0.025       0       0
    1003            0.02           0.015       0.0249781       0       0
    1004       0.0198317       0.0162864       0.0249878       0       0
    1005            0.02           0.013       0.0249781       0       0
    1006           0.025           0.008       0.0232427       0       0
    1007       0.0235958       0.0082014       0.0240481       0       0
    1008        0.022202       0.0088562       0.0245811       0       0
    1009       0.0210325        0.009957       0.0248557       0       0
    1010       0.0202656       0.0113917       0.0249572       0       0
    1012       0.0261157           0.008       0.0223528       0       0
    1013       0.0270933           0.008       0.0213132       0       0
    1014        0.027913         0.00625        0.020145       0       0
    1015       0.0204079       0.0058219       0.0249426       0       0
    1016       0.0219254       0.0067136       0.0246595       0       0
    1017       0.0270918       0.0062503       0.0213151       0       0
    1018       0.0261061       0.0062562       0.0223617       0       0
    1019        0.024918       0.0063005       0.0232985       0       0
    1020       0.0235334       0.0064463       0.0240774       0       0
    1021       0.0205178       0.0077989       0.0249302       0       0
    1024      -0.0170232    4.999998e-04           0.025       0       0
    1028      -0.0181469       0.0030606           0.025       0       0
    1031      -0.0190636   -5.000000e-04       0.0258007       0       0
    1036      -0.0181191    5.000000e-04       0.0252074       0       0
    1039      -0.0193381      -0.0038524           0.025       0       0
    1040      -0.0198292      -0.0027546       0.0252516       0       0
    1041      -0.0170232   -5.000001e-04           0.025       0       0
    1042      -0.0181469      -0.0030606           0.025       0       0
    1043      -0.0173208      -0.0018955           0.025       0       0
    1045      -0.0181191   -5.000001e-04       0.0252074       0       0
    1050      -0.0218157         -0.0045       0.0246882       0       0
    1051      -0.0234715         -0.0045       0.0241058       0       0
    1055      -0.0193381      -0.0174861           0.025       0       0
    1059      -0.0193381      -0.0096954           0.025       0       0
    1062           -0.02          -0.015       0.0249781       0       0
    1063      -0.0198317      -0.0162864       0.0249878       0       0
    1064           -0.02          -0.013       0.0249781       0       0
    1065          -0.025          -0.008       0.0232427       0       0
    1066      -0.0235958      -0.0082014       0.0240481       0       0
    1067       -0.022202      -0.0088562       0.0245811       0       0
    1068      -0.0210325       -0.009957       0.0248557       0       0
    1069      -0.0202656      -0.0113917       0.0249572       0       0
    1070       -0.027913          -0.008        0.020145       0       0
    1071      -0.0261157          -0.008       0.0223528       0       0
    1072      -0.0270933          -0.008       0.0213132       0       0
    1074      -0.0204079      -0.0058219       0.0249426       0       0
    1075      -0.0219254      -0.0067136       0.0246595       0       0
    1076      -0.0270918      -0.0062503       0.0213151       0       0
    1077       -0.026106      -0.0062562       0.0223617       0       0
    1078       -0.024918      -0.0063005       0.0232985       0       0
    1079      -0.0235334      -0.0064463       0.0240774       0       0
    1080      -0.0205178      -0.0077989       0.0249302       0       0
    1083      -0.0043409      -0.0205439        0.024927       0       0
    1087         -0.0045       -0.027913        0.020145       0       0
    1090         -0.0045      -0.0261157       0.0223529       0       0
    1091          -0.008       -0.027913        0.020145       0       0
    1092        -0.00625       -0.027913        0.020145       0       0
    1093          -0.008          -0.025       0.0232427       0       0
    1094          -0.008      -0.0261157       0.0223528       0       0
    1095          -0.008      -0.0270933       0.0213132       0       0
    1096          -0.013           -0.02       0.0249781       0       0
    1097      -0.0113917      -0.0202656       0.0249572       0       0
    1098       -0.009957      -0.0210325       0.0248557       0       0
    1099      -0.0088562       -0.022202       0.0245811       0       0
    1100      -0.0082014      -0.0235958       0.0240481       0       0
    1101          -0.015           -0.02       0.0249781       0       0
    1103      -0.0162864      -0.0198317       0.0249878       0       0
    1110      -0.0058219      -0.0204079       0.0249426       0       0
    1111      -0.0067136      -0.0219254       0.0246595       0       0
    1112      -0.0062503      -0.0270918       0.0213151       0       0
    1113      -0.0062562       -0.026106       0.0223617       0       0
    1114      -0.0063005       -0.024918       0.0232985       0       0
    1115      -0.0064463      -0.0235334       0.0240774       0       0
    1116      -0.0077989      -0.0205178       0.0249302       0       0
    1118      -0.0038524      -0.0193381           0.025       0       0
    1119      -0.0018955      -0.0173208           0.025       0       0
    1122      -0.0027546      -0.0198292       0.0252516       0       0
    1124   -5.000001e-04      -0.0181191       0.0252074       0       0
    1127   -5.000000e-04      -0.0190636       0.0258007       0       0
    1131       0.0038524      -0.0193381           0.025       0       0
    1132    4.999998e-04      -0.0170232           0.025       0       0
    1134       0.0018955      -0.0173208           0.025       0       0
    1135    5.000000e-04      -0.0190636       0.0258007       0       0
    1136    5.000000e-04      -0.0181191       0.0252074       0       0
    1137       0.0018409      -0.0202379       0.0259642       0       0
    1141          0.0045      -0.0261157       0.0223528       0       0
    1145          0.0045      -0.0234715       0.0241058       0       0
    1149       0.0155385      -0.0193381           0.025       0       0
    1150       0.0135908      -0.0193381           0.025       0       0
    1151       0.0116431      -0.0193381           0.025       0       0
    1153       0.0077477      -0.0193381           0.025       0       0
    1154          0.0058      -0.0193381           0.025       0       0
    1155           0.015           -0.02       0.0249781       0       0
    1156       0.0162864      -0.0198317       0.0249878       0       0
    1157           0.013           -0.02       0.0249781       0       0
    1158           0.008          -0.025       0.0232427       0       0
    1159       0.0082014      -0.0235958       0.0240481       0       0
    1160       0.0088562       -0.022202       0.0245811       0       0
    1161        0.009957      -0.0210325       0.0248557       0       0
    1162       0.0113917      -0.0202656       0.0249572       0       0
    1163           0.008       -0.027913        0.020145       0       0
    1164           0.008      -0.0261157       0.0223528       0       0
    1165           0.008      -0.0270933       0.0213132       0       0
    1166         0.00625       -0.027913        0.020145       0       0
    1167       0.0058219      -0.0204079       0.0249426       0       0
    1168       0.0067136      -0.0219254       0.0246595       0       0
    1169       0.0062503      -0.0270918       0.0213151       0       0
    1170       0.0062562      -0.0261061       0.0223617       0       0
    1171       0.0063005       -0.024918       0.0232985       0       0
    1172       0.0064463      -0.0235334       0.0240774       0       0
    1173       0.0077989      -0.0205178       0.0249302       0       0
    1180        0.027913         -0.0045        0.020145       0       0
    1182       0.0270933         -0.0045       0.0213132       0       0
    1183       0.0261157         -0.0045       0.0223529       0       0
    1186           0.025          -0.008       0.0232427       0       0
    1187       0.0261157          -0.008       0.0223528       0       0
    1188       0.0270933          -0.008       0.0213132       0       0
    1189            0.02          -0.013       0.0249781       0       0
    1190       0.0202656      -0.0113917       0.0249572       0       0
    1191       0.0210325       -0.009957       0.0248557       0       0
    1192        0.022202      -0.0088562       0.0245811       0       0
    1193       0.0235958      -0.0082014       0.0240481       0       0
    1194            0.02          -0.015       0.0249781       0       0
    1196       0.0198317      -0.0162864       0.0249878       0       0
    1202       0.0193381      -0.0155385           0.025       0       0
    1203       0.0204079      -0.0058219       0.0249426       0       0
    1204       0.0219254      -0.0067136       0.0246595       0       0
    1205       0.0270918      -0.0062503       0.0213151       0       0
    1206        0.026106      -0.0062562       0.0223617       0       0
    1207        0.024918      -0.0063005       0.0232985       0       0
    1208       0.0235334      -0.0064463       0.0240774       0       0
    1209       0.0205178      -0.0077989       0.0249302       0       0
    1211       0.0193381      -0.0038524           0.025       0       0
    1216       0.0190636   -5.000001e-04       0.0258007       0       0
    1217       0.0181191   -5.000001e-04       0.0252074       0       0
    1226       0.0181469       0.0030606           0.025       0       0
    1227       0.0173208       0.0018955           0.025       0       0
    1228       0.0190636    5.000000e-04       0.0258007       0       0
    1229       0.0181191    5.000000e-04       0.0252074       0       0
    1233       -0.037087           0.008        0.004855       0       0
    1236      -0.0289324           0.008       0.0184461       0       0
    1237      -0.0299517           0.008       0.0167472       0       0
    1238       -0.030971           0.008       0.0150483       0       0
    1239      -0.0319903           0.008       0.0133494       0       0
    1240      -0.0330097           0.008       0.0116506       0       0
    1241       -0.034029           0.008       0.0099517       0       0
    1242      -0.0350483           0.008       0.0082528       0       0
    1243      -0.0360676           0.008       0.0065539       0       0
    1246      -0.0289324          0.0045       0.0184461       0       0
    1247      -0.0299517          0.0045       0.0167472       0       0
    1249      -0.0319903          0.0045       0.0133494       0       0
    1254      -0.0360676       0.0062496       0.0065539       0       0
    1255      -0.0289324       0.0062499       0.0184461       0       0
    1256      -0.0350483       0.0062472       0.0082528       0       0
    1257      -0.0299517       0.0062493       0.0167472       0       0
    1258       -0.030971       0.0062492       0.0150483       0       0
    1259      -0.0319903       0.0062484       0.0133494       0       0
    1260       -0.034029       0.0062487       0.0099517       0       0
    1261      -0.0330097       0.0062496       0.0116506       0       0
    1263      -0.0256794          0.0021       0.0242319       0       0
    1267      -0.0251793       0.0031584       0.0235038       0       0
    1268       -0.027913          0.0045        0.020145       0       0
    1269      -0.0261157          0.0045       0.0223528       0       0
    1270      -0.0270933          0.0045       0.0213132       0       0
    1272       -0.026831        0.003223        0.022046       0       0
    1273       -0.022113          0.0021       0.0258508       0       0
    1275      -0.0239675          0.0021       0.0251984       0       0
    1285      -0.0235922       0.0032091       0.0243716       0       0
    1286      -0.0202379       0.0018409       0.0259642       0       0
    1289      -0.0198292       0.0027546       0.0252516       0       0
    1293      -0.0218942       0.0031583       0.0249951       0       0
    1294        0.027913           0.008        0.020145       0       0
    1297        0.037087           0.008        0.004855       0       0
    1298       0.0289324           0.008       0.0184461       0       0
    1299       0.0299517           0.008       0.0167472       0       0
    1300        0.030971           0.008       0.0150483       0       0
    1301       0.0319903           0.008       0.0133494       0       0
    1302       0.0330097           0.008       0.0116506       0       0
    1303        0.034029           0.008       0.0099517       0       0
    1304       0.0350483           0.008       0.0082528       0       0
    1305       0.0360676           0.008       0.0065539       0       0
    1308       0.0360676          0.0045       0.0065539       0       0
    1309       0.0350483          0.0045       0.0082528       0       0
    1310        0.034029          0.0045       0.0099517       0       0
    1311       0.0330097          0.0045       0.0116506       0       0
    1312       0.0319903          0.0045       0.0133494       0       0
    1313        0.030971          0.0045       0.0150483       0       0
    1314       0.0299517          0.0045       0.0167472       0       0
    1315       0.0289324          0.0045       0.0184461       0       0
    1316       0.0289324       0.0062501       0.0184461       0       0
    1317       0.0360676       0.0062504       0.0065539       0       0
    1318       0.0350483       0.0062528       0.0082528       0       0
    1319       0.0299517       0.0062507       0.0167472       0       0
    1320        0.030971       0.0062508       0.0150483       0       0
    1321       0.0319903       0.0062516       0.0133494       0       0
    1322        0.034029       0.0062513       0.0099517       0       0
    1323       0.0330097       0.0062504       0.0116506       0       0
    1327       0.0202379       0.0018409       0.0259642       0       0
    1328       0.0198292       0.0027546       0.0252516       0       0
    1329        0.022113          0.0021       0.0258508       0       0
    1330       0.0218942       0.0031583       0.0249951       0       0
    1333       0.0251793       0.0031584       0.0235038       0       0
    1342       0.0239675          0.0021       0.0251984       0       0
    1343       0.0235922       0.0032091       0.0243716       0       0
    1348       0.0256794          0.0021       0.0242319       0       0
    1350        0.028942          0.0021       0.0207624       0       0
    1352        0.028024          0.0021       0.0220708       0       0
    1353        0.026929          0.0021       0.0232352       0       0
    1355        0.026831        0.003223        0.022046       0       0
    1358      -0.0190636    5.000000e-04       0.0258007       0       0
    1360      -0.0205822    5.000000e-04       0.0266272       0       0
    1371      -0.0218942      -0.0031583       0.0249951       0       0
    1373      -0.0205439      -0.0043409        0.024927       0       0
    1375      -0.0202379      -0.0018409       0.0259642       0       0
    1381      -0.0205822   -5.000000e-04       0.0266272       0       0
    1384       -0.027913        -0.00625        0.020145       0       0
    1386      -0.0289324          -0.008       0.0184461       0       0
    1387      -0.0299517          -0.008       0.0167472       0       0
    1388       -0.030971          -0.008       0.0150483       0       0
    1389      -0.0319903          -0.008       0.0133494       0       0
    1390      -0.0330097          -0.008       0.0116506       0       0
    1391       -0.034029          -0.008       0.0099517       0       0
    1392      -0.0350483          -0.008       0.0082528       0       0
    1393      -0.0360676          -0.008       0.0065539       0       0
    1404      -0.0289324      -0.0062501       0.0184461       0       0
    1405      -0.0360676      -0.0062504       0.0065539       0       0
    1406      -0.0350483      -0.0062528       0.0082528       0       0
    1407      -0.0299517      -0.0062507       0.0167472       0       0
    1408       -0.030971      -0.0062508       0.0150483       0       0
    1409      -0.0319903      -0.0062516       0.0133494       0       0
    1410       -0.034029      -0.0062513       0.0099517       0       0
    1411      -0.0330097      -0.0062504       0.0116506       0       0
    1413       -0.022113         -0.0021       0.0258508       0       0
    1415          -0.025         -0.0045       0.0232428       0       0
    1416      -0.0251793      -0.0031584       0.0235038       0       0
    1422      -0.0235922      -0.0032091       0.0243716       0       0
    1428      -0.0281846      -0.0031584       0.0203079       0       0
    1430      -0.0270933         -0.0045       0.0213132       0       0
    1431      -0.0261157         -0.0045       0.0223529       0       0
    1435       -0.026831       -0.003223        0.022046       0       0
    1436         -0.0045       -0.037087        0.004855       0       0
    1440          -0.008      -0.0289324       0.0184461       0       0
    1441          -0.008      -0.0299517       0.0167472       0       0
    1442          -0.008       -0.030971       0.0150483       0       0
    1443          -0.008      -0.0319903       0.0133494       0       0
    1444          -0.008      -0.0330097       0.0116506       0       0
    1445          -0.008       -0.034029       0.0099517       0       0
    1446          -0.008      -0.0350483       0.0082528       0       0
    1447          -0.008      -0.0360676       0.0065539       0       0
    1452         -0.0045       -0.030971       0.0150483       0       0
    1454         -0.0045      -0.0330097       0.0116506       0       0
    1455         -0.0045       -0.034029       0.0099517       0       0
    1456         -0.0045      -0.0350483       0.0082528       0       0
    1458      -0.0062496      -0.0360676       0.0065539       0       0
    1459      -0.0062499      -0.0289324       0.0184461       0       0
    1460      -0.0062472      -0.0350483       0.0082528       0       0
    1461      -0.0062493      -0.0299517       0.0167472       0       0
    1462      -0.0062492       -0.030971       0.0150483       0       0
    1463      -0.0062484      -0.0319903       0.0133494       0       0
    1464      -0.0062487       -0.034029       0.0099517       0       0
    1465      -0.0062496      -0.0330097       0.0116506       0       0
    1467         -0.0045          -0.025       0.0232428       0       0
    1468         -0.0045      -0.0270933       0.0213132       0       0
    1471      -0.0031584      -0.0281846       0.0203079       0       0
    1472         -0.0021      -0.0256794       0.0242319       0       0
    1477      -0.0031584      -0.0251793       0.0235038       0       0
    1478       -0.003223       -0.026831        0.022046       0       0
    1481         -0.0045      -0.0234715       0.0241058       0       0
    1486      -0.0031583      -0.0218942       0.0249951       0       0
    1487      -0.0032091      -0.0235922       0.0243716       0       0
    1488         -0.0045      -0.0218157       0.0246882       0       0
    1491         -0.0021       -0.022113       0.0258508       0       0
    1493      -0.0018409      -0.0202379       0.0259642       0       0
    1501   -5.000000e-04      -0.0205822       0.0266272       0       0
    1505    5.000000e-04      -0.0205822       0.0266272       0       0
    1510       0.0043409      -0.0205439        0.024927       0       0
    1512       0.0027546      -0.0198292       0.0252516       0       0
    1514       0.0031583      -0.0218942       0.0249951       0       0
    1523           0.008       -0.037087        0.004855       0       0
    1524           0.008      -0.0289324       0.0184461       0       0
    1525           0.008      -0.0299517       0.0167472       0       0
    1526           0.008       -0.030971       0.0150483       0       0
    1527           0.008      -0.0319903       0.0133494       0       0
    1528           0.008      -0.0330097       0.0116506       0       0
    1529           0.008       -0.034029       0.0099517       0       0
    1530           0.008      -0.0350483       0.0082528       0       0
    1531           0.008      -0.0360676       0.0065539       0       0
    1532          0.0045       -0.037087        0.004855       0       0
    1533         0.00625       -0.037087        0.004855       0       0
    1534          0.0045      -0.0360676       0.0065539       0       0
    1536          0.0045       -0.034029       0.0099517       0       0
    1540          0.0045      -0.0299517       0.0167472       0       0
    1541          0.0045      -0.0289324       0.0184461       0       0
    1542       0.0062501      -0.0289324       0.0184461       0       0
    1543       0.0062504      -0.0360676       0.0065539       0       0
    1544       0.0062528      -0.0350483       0.0082528       0       0
    1545       0.0062507      -0.0299517       0.0167472       0       0
    1546       0.0062508       -0.030971       0.0150483       0       0
    1547       0.0062516      -0.0319903       0.0133494       0       0
    1548       0.0062513       -0.034029       0.0099517       0       0
    1549       0.0062504      -0.0330097       0.0116506       0       0
    1550          0.0021      -0.0256794       0.0242319       0       0
    1553          0.0045      -0.0218157       0.0246882       0       0
    1556          0.0021       -0.022113       0.0258508       0       0
    1562       0.0032091      -0.0235922       0.0243716       0       0
    1563          0.0045          -0.025       0.0232428       0       0
    1564          0.0045       -0.027913        0.020145       0       0
    1566          0.0045      -0.0270933       0.0213132       0       0
    1568       0.0031584      -0.0251793       0.0235038       0       0
    1572          0.0021       -0.026929       0.0232352       0       0
    1574        0.003223       -0.026831        0.022046       0       0
    1578        0.027913          -0.008        0.020145       0       0
    1579       0.0289324          -0.008       0.0184461       0       0
    1580       0.0299517          -0.008       0.0167472       0       0
    1581        0.030971          -0.008       0.0150483       0       0
    1582       0.0319903          -0.008       0.0133494       0       0
    1583       0.0330097          -0.008       0.0116506       0       0
    1584        0.034029          -0.008       0.0099517       0       0
    1585       0.0350483          -0.008       0.0082528       0       0
    1586       0.0360676          -0.008       0.0065539       0       0
    1588        0.027913        -0.00625        0.020145       0       0
    1597       0.0360676      -0.0062496       0.0065539       0       0
    1598       0.0289324      -0.0062499       0.0184461       0       0
    1599       0.0350483      -0.0062472       0.0082528       0       0
    1600       0.0299517      -0.0062493       0.0167472       0       0
    1601        0.030971      -0.0062492       0.0150483       0       0
    1602       0.0319903      -0.0062484       0.0133494       0       0
    1603        0.034029      -0.0062487       0.0099517       0       0
    1604       0.0330097      -0.0062496       0.0116506       0       0
    1610       0.0281846      -0.0031584       0.0203079       0       0
    1611       0.0256794         -0.0021       0.0242319       0       0
    1617        0.026831       -0.003223        0.022046       0       0
    1618       0.0218157         -0.0045       0.0246882       0       0
    1621           0.025         -0.0045       0.0232428       0       0
    1623       0.0234715         -0.0045       0.0241058       0       0
    1625       0.0251793      -0.0031584       0.0235038       0       0
    1627       0.0239675         -0.0021       0.0251984       0       0
    1628       0.0235922      -0.0032091       0.0243716       0       0
    1631       0.0205439      -0.0043409        0.024927       0       0
    1632        0.022113         -0.0021       0.0258508       0       0
    1633       0.0218942      -0.0031583       0.0249951       0       0
    1635       0.0198292      -0.0027546       0.0252516       0       0
    1636       0.0202379      -0.0018409       0.0259642       0       0
    1646       0.0205822    5.000000e-04       0.0266272       0       0
    1647       0.0223112   -5.000000e-04       0.0266258       0       0
    1648       0.0205822   -5.000000e-04       0.0266272       0       0
    1654      -0.0456619           0.008             0.0       0       0
    1657          -0.057           0.008             0.0       0       0
    1658      -0.0475516           0.008             0.0       0       0
    1659      -0.0494413           0.008             0.0       0       0
    1660       -0.051331           0.008             0.0       0       0
    1661      -0.0532206           0.008             0.0       0       0
    1662      -0.0551103           0.008             0.0       0       0
    1663           -0.06           0.005             0.0       0       0
    1664         -0.0585       0.0075981             0.0       0       0
    1665      -0.0595981          0.0065             0.0       0       0
    1666           -0.06          -0.005             0.0       0       0
    1667           -0.06          -0.003             0.0       0       0
    1668           -0.06          -0.001             0.0       0       0
    1669           -0.06           0.001             0.0       0       0
    1670           -0.06           0.003             0.0       0       0
    1671          -0.057          -0.008             0.0       0       0
    1672         -0.0585      -0.0075981             0.0       0       0
    1673      -0.0595981         -0.0065             0.0       0       0
    1675      -0.0551103          -0.008             0.0       0       0
    1676      -0.0532206          -0.008             0.0       0       0
    1677       -0.051331          -0.008             0.0       0       0
    1678      -0.0494413          -0.008             0.0       0       0
    1679      -0.0475516          -0.008             0.0       0       0
    1680      -0.0456619      -0.0040694   -2.441217e-09       0       0
    1681      -0.0456619      -0.0060347             0.0       0       0
    1682      -0.0478798   -4.999999e-04    5.294654e-16       0       0
    1683      -0.0476024      -0.0019359             0.0       0       0
    1684      -0.0468199      -0.0031724             0.0       0       0
    1688          -0.056             0.0             0.0       0       0
    1689           -0.05    4.592426e-20             0.0       0       0
    1690      -0.0554271      -0.0017634             0.0       0       0
    1691      -0.0539271      -0.0028532             0.0       0       0
    1692      -0.0520729      -0.0028532             0.0       0       0
    1693       -0.050573      -0.0017634             0.0       0       0
    1694       -0.050573       0.0017634             0.0       0       0
    1695      -0.0520729       0.0028532             0.0       0       0
    1696      -0.0539271       0.0028532             0.0       0       0
    1697      -0.0554271       0.0017634             0.0       0       0
    1698      -0.0473859       0.0058995             0.0       0       0
    1699      -0.0473843       -0.005933             0.0       0       0
    1700      -0.0490107       0.0060553             0.0       0       0
    1702      -0.0492708      -0.0059352             0.0       0       0
    1703      -0.0512862      -0.0060536             0.0       0       0
    1704      -0.0532916       0.0058767             0.0       0       0
    1705      -0.0536999      -0.0058379             0.0       0       0
    1706      -0.0551529       0.0060091             0.0       0       0
    1707      -0.0552647      -0.0067009             0.0       0       0
    1708      -0.0571712       0.0060852             0.0       0       0
    1709      -0.0570384      -0.0059644             0.0       0       0
    1710       -0.049005      -0.0042525             0.0       0       0
    1711      -0.0493286      -0.0026496             0.0       0       0
    1712      -0.0492547       0.0038749             0.0       0       0
    1714      -0.0512908       0.0054399             0.0       0       0
    1715      -0.0534096       0.0045342             0.0       0       0
    1716      -0.0547173        0.004172             0.0       0       0
    1721       -0.057286       0.0034014             0.0       0       0
    1722      -0.0581233       0.0010309             0.0       0       0
    1723      -0.0581414   -9.557459e-04             0.0       0       0
    1724      -0.0574303      -0.0029579             0.0       0       0
    1725      -0.0582618      -0.0046366             0.0       0       0
    1726      -0.0555025      -0.0043857             0.0       0       0
    1728      -0.0513483      -0.0040616             0.0       0       0
    1729      -0.0529727      -0.0043416             0.0       0       0
    1731      -0.0456619       0.0040693   -9.313226e-10       0       0
    1733       -0.037087          0.0045        0.004855       0       0
    1734      -0.0382573          0.0045        0.003279       0       0
    1735       -0.039713          0.0045       0.0019619       0       0
    1736       -0.041398          0.0045    9.546103e-04       0       0
    1738       -0.037087         0.00625        0.004855       0       0
    1740      -0.0380923           0.008       0.0034654       0       0
    1741      -0.0393203           0.008        0.002268       0       0
    1742      -0.0407349           0.008        0.001298       0       0
    1743      -0.0422944           0.008    5.840436e-04       0       0
    1744       -0.043953           0.008    1.470922e-04       0       0
    1745      -0.0456619       0.0060347   -9.313226e-10       0       0
    1748      -0.0381739       0.0062499        0.003372       0       0
    1749      -0.0395188       0.0062524       0.0021094       0       0
    1750      -0.0411025       0.0062697       0.0010999       0       0
    1751      -0.0431248        0.006411    3.271997e-04       0       0
    1755      -0.0309807          0.0021       0.0173646       0       0
    1756          -0.032          0.0021       0.0156657       0       0
    1757      -0.0330193          0.0021       0.0139668       0       0
    1759       -0.035058          0.0021       0.0105691       0       0
    1763      -0.0281846       0.0031584       0.0203079       0       0
    1767       -0.030971          0.0045       0.0150483       0       0
    1769      -0.0330097          0.0045       0.0116506       0       0
    1770       -0.034029          0.0045       0.0099517       0       0
    1771      -0.0350483          0.0045       0.0082528       0       0
    1772      -0.0360676          0.0045       0.0065539       0       0
    1773      -0.0373586       0.0031584        0.005018       0       0
    1776       -0.028942          0.0021       0.0207624       0       0
    1778       -0.028024          0.0021       0.0220708       0       0
    1779       -0.026929          0.0021       0.0232352       0       0
    1783      -0.0286444    5.000000e-04       0.0225758       0       0
    1787      -0.0223112    5.000000e-04       0.0266258       0       0
    1788      -0.0261324    5.000015e-04       0.0248913       0       0
    1789      -0.0236499    5.000000e-04       0.0261985       0       0
    1790      -0.0249295    5.000000e-04       0.0256177       0       0
    1793        0.037087          0.0045        0.004855       0       0
    1795       0.0382573          0.0045        0.003279       0       0
    1796        0.039713          0.0045       0.0019619       0       0
    1801       0.0456619           0.008   -9.313226e-10       0       0
    1802       0.0456619       0.0060347   -9.313226e-10       0       0
    1804        0.043953           0.008    1.470922e-04       0       0
    1805       0.0422944           0.008    5.840436e-04       0       0
    1806       0.0407349           0.008        0.001298       0       0
    1807       0.0393203           0.008        0.002268       0       0
    1808       0.0380923           0.008       0.0034654       0       0
    1809        0.037087         0.00625        0.004855       0       0
    1811        0.043123       0.0064102    3.276709e-04       0       0
    1812       0.0381743       0.0062501       0.0033716       0       0
    1813        0.039519       0.0062525       0.0021092       0       0
    1814       0.0411038       0.0062701       0.0010992       0       0
    1827       0.0281846       0.0031584       0.0203079       0       0
    1829       0.0370966          0.0021       0.0071713       0       0
    1830       0.0360773          0.0021       0.0088702       0       0
    1831        0.035058          0.0021       0.0105691       0       0
    1832       0.0340387          0.0021        0.012268       0       0
    1833       0.0330193          0.0021       0.0139668       0       0
    1841       0.0456619      -0.0040694   -2.458484e-09       0       0
    1842       0.0456619      -0.0060347             0.0       0       0
    1843           0.057          -0.008             0.0       0       0
    1844       0.0475516          -0.008             0.0       0       0
    1845       0.0494413          -0.008             0.0       0       0
    1846        0.051331          -0.008             0.0       0       0
    1847       0.0532206          -0.008             0.0       0       0
    1848       0.0551103          -0.008             0.0       0       0
    1849            0.06          -0.005             0.0       0       0
    1850       0.0595981         -0.0065             0.0       0       0
    1851          0.0585      -0.0075981             0.0       0       0
    1852            0.06           0.005             0.0       0       0
    1853            0.06           0.003             0.0       0       0
    1854            0.06           0.001             0.0       0       0
    1855            0.06          -0.001             0.0       0       0
    1856            0.06          -0.003             0.0       0       0
    1857           0.057           0.008             0.0       0       0
    1858          0.0585       0.0075981             0.0       0       0
    1859       0.0595981          0.0065             0.0       0       0
    1861       0.0551103           0.008             0.0       0       0
    1862       0.0532206           0.008             0.0       0       0
    1863        0.051331           0.008             0.0       0       0
    1864       0.0494413           0.008             0.0       0       0
    1865       0.0475516           0.008             0.0       0       0
    1871       0.0478798   -4.999999e-04    2.785550e-16       0       0
    1872       0.0468199      -0.0031724             0.0       0       0
    1873       0.0476024      -0.0019359             0.0       0       0
    1874           0.056             0.0             0.0       0       0
    1875            0.05   -4.592426e-20             0.0       0       0
    1876       0.0554271       0.0017634             0.0       0       0
    1877       0.0539271       0.0028532             0.0       0       0
    1878       0.0520729       0.0028532             0.0       0       0
    1879        0.050573       0.0017634             0.0       0       0
    1880        0.050573      -0.0017634             0.0       0       0
    1881       0.0520729      -0.0028532             0.0       0       0
    1882       0.0539271      -0.0028532             0.0       0       0
    1883       0.0554271      -0.0017634             0.0       0       0
    1884       0.0472682      -0.0058946             0.0       0       0
    1885       0.0473859       0.0058995             0.0       0       0
    1886        0.048881      -0.0057038             0.0       0       0
    1888       0.0490107       0.0060553             0.0       0       0
    1890       0.0532778      -0.0059701             0.0       0       0
    1891       0.0532916       0.0058767             0.0       0       0
    1892       0.0549954      -0.0061826             0.0       0       0
    1893       0.0551529       0.0060091             0.0       0       0
    1894       0.0571514      -0.0057868             0.0       0       0
    1895       0.0571712       0.0060852             0.0       0       0
    1896       0.0484577      -0.0041818             0.0       0       0
    1897       0.0491636      -0.0031646             0.0       0       0
    1898       0.0492547       0.0038749             0.0       0       0
    1900       0.0512908       0.0054399             0.0       0       0
    1901       0.0534096       0.0045342             0.0       0       0
    1902       0.0547173        0.004172             0.0       0       0
    1907        0.057286       0.0034014             0.0       0       0
    1908       0.0581235       0.0010321             0.0       0       0
    1909       0.0581464   -9.468379e-04             0.0       0       0
    1910       0.0574702      -0.0029076             0.0       0       0
    1911        0.058324      -0.0045615             0.0       0       0
    1912       0.0510105      -0.0052236             0.0       0       0
    1913       0.0536858      -0.0045995             0.0       0       0
    1914       0.0547932      -0.0049311             0.0       0       0
    1915       0.0557221      -0.0041943             0.0       0       0
    1919       0.0261324    5.000000e-04       0.0248913       0       0
    1921       0.0223112    5.000000e-04       0.0266258       0       0
    1923       0.0236499    5.000000e-04       0.0261985       0       0
    1924       0.0249295    5.000000e-04       0.0256177       0       0
    1931       0.0286444    5.000000e-04       0.0225758       0       0
    1932       0.0274712    5.000000e-04       0.0238234       0       0
    1938      -0.0223112   -5.000000e-04       0.0266258       0       0
    1939      -0.0261324   -5.000000e-04       0.0248913       0       0
    1940      -0.0236499   -5.000000e-04       0.0261985       0       0
    1941      -0.0249295   -5.000000e-04       0.0256177       0       0
    1942      -0.0256794         -0.0021       0.0242319       0       0
    1944      -0.0239675         -0.0021       0.0251984       0       0
    1952      -0.0432473         -0.0045    2.959035e-04       0       0
    1953      -0.0382573         -0.0045        0.003279       0       0
    1954       -0.039713         -0.0045       0.0019619       0       0
    1955       -0.041398         -0.0045    9.546094e-04       0       0
    1958      -0.0456619          -0.008   -9.313226e-10       0       0
    1960       -0.037087          -0.008        0.004855       0       0
    1961       -0.043953          -0.008    1.470922e-04       0       0
    1962      -0.0422944          -0.008    5.840436e-04       0       0
    1963      -0.0407349          -0.008        0.001298       0       0
    1964      -0.0393203          -0.008        0.002268       0       0
    1965      -0.0380923          -0.008       0.0034654       0       0
    1966       -0.037087        -0.00625        0.004855       0       0
    1968       -0.043123      -0.0064102    3.276709e-04       0       0
    1969      -0.0381743      -0.0062501       0.0033716       0       0
    1970       -0.039519      -0.0062525       0.0021092       0       0
    1971      -0.0411038      -0.0062701       0.0010992       0       0
    1973       -0.038116         -0.0021       0.0054724       0       0
    1982      -0.0299613         -0.0021       0.0190635       0       0
    1983       -0.037087         -0.0045        0.004855       0       0
    1985       -0.027913         -0.0045        0.020145       0       0
    1986      -0.0360676         -0.0045       0.0065539       0       0
    1987      -0.0350483         -0.0045       0.0082528       0       0
    1988       -0.034029         -0.0045       0.0099517       0       0
    1989      -0.0330097         -0.0045       0.0116506       0       0
    1990      -0.0319903         -0.0045       0.0133494       0       0
    1991       -0.030971         -0.0045       0.0150483       0       0
    1992      -0.0299517         -0.0045       0.0167472       0       0
    1993      -0.0289324         -0.0045       0.0184461       0       0
    1997       -0.028942         -0.0021       0.0207624       0       0
    1999       -0.028024         -0.0021       0.0220708       0       0
    2000       -0.026929         -0.0021       0.0232352       0       0
    2003      -0.0286444   -5.000000e-04       0.0225758       0       0
    2004      -0.0274712   -5.000000e-04       0.0238234       0       0
    2008       -0.004395      -0.0444651    7.187016e-05       0       0
    2011         -0.0045       -0.039713       0.0019619       0       0
    2013          -0.008       -0.037087        0.004855       0       0
    2014        -0.00625       -0.037087        0.004855       0       0
    2016          -0.008      -0.0380923       0.0034654       0       0
    2017          -0.008      -0.0393203        0.002268       0       0
    2018          -0.008      -0.0407349        0.001298       0       0
    2019          -0.008      -0.0422944    5.840436e-04       0       0
    2020          -0.008       -0.043953    1.470922e-04       0       0
    2024      -0.0062499      -0.0381739        0.003372       0       0
    2025      -0.0062524      -0.0395188       0.0021094       0       0
    2026      -0.0062697      -0.0411025       0.0010999       0       0
    2027       -0.006411      -0.0431248    3.271997e-04       0       0
    2029          -0.008      -0.0456619             0.0       0       0
    2030      -0.0060347      -0.0456619             0.0       0       0
    2031          -0.008          -0.057             0.0       0       0
    2032          -0.008      -0.0475516             0.0       0       0
    2033          -0.008      -0.0494413             0.0       0       0
    2034          -0.008       -0.051331             0.0       0       0
    2035          -0.008      -0.0532206             0.0       0       0
    2036          -0.008      -0.0551103             0.0       0       0
    2037          -0.005           -0.06             0.0       0       0
    2038         -0.0065      -0.0595981             0.0       0       0
    2039      -0.0075981         -0.0585             0.0       0       0
    2040           0.005           -0.06             0.0       0       0
    2041           0.003           -0.06             0.0       0       0
    2042           0.001           -0.06             0.0       0       0
    2043          -0.001           -0.06             0.0       0       0
    2044          -0.003           -0.06             0.0       0       0
    2045           0.008          -0.057             0.0       0       0
    2046       0.0075981         -0.0585             0.0       0       0
    2047          0.0065      -0.0595981             0.0       0       0
    2049           0.008      -0.0551103             0.0       0       0
    2050           0.008      -0.0532206             0.0       0       0
    2051           0.008       -0.051331             0.0       0       0
    2052           0.008      -0.0494413             0.0       0       0
    2053           0.008      -0.0475516             0.0       0       0
    2054       0.0040693      -0.0456619             0.0       0       0
    2055       0.0060347      -0.0456619             0.0       0       0
    2057       0.0031724      -0.0468199             0.0       0       0
    2058       0.0019359      -0.0476024             0.0       0       0
    2060      -0.0031724      -0.0468199             0.0       0       0
    2061      -0.0019359      -0.0476024             0.0       0       0
    2062             0.0          -0.056             0.0       0       0
    2063   -4.592426e-20           -0.05             0.0       0       0
    2064       0.0017634      -0.0554271             0.0       0       0
    2065       0.0028532      -0.0539271             0.0       0       0
    2066       0.0028532      -0.0520729             0.0       0       0
    2067       0.0017634       -0.050573             0.0       0       0
    2068      -0.0017634       -0.050573             0.0       0       0
    2069      -0.0028532      -0.0520729             0.0       0       0
    2070      -0.0028532      -0.0539271             0.0       0       0
    2071      -0.0017634      -0.0554271             0.0       0       0
    2072      -0.0058946      -0.0472682             0.0       0       0
    2073       0.0058995      -0.0473859             0.0       0       0
    2074      -0.0057037       -0.048881             0.0       0       0
    2076       0.0060553      -0.0490107             0.0       0       0
    2078      -0.0059701      -0.0532778             0.0       0       0
    2079       0.0058767      -0.0532916             0.0       0       0
    2080      -0.0061826      -0.0549954             0.0       0       0
    2081       0.0060091      -0.0551529             0.0       0       0
    2082      -0.0057868      -0.0571514             0.0       0       0
    2083       0.0060852      -0.0571712             0.0       0       0
    2084      -0.0041818      -0.0484577             0.0       0       0
    2085      -0.0031646      -0.0491636             0.0       0       0
    2086       0.0038749      -0.0492547             0.0       0       0
    2088       0.0054399      -0.0512908             0.0       0       0
    2089       0.0045342      -0.0534096             0.0       0       0
    2090        0.004172      -0.0547173             0.0       0       0
    2095       0.0034014       -0.057286             0.0       0       0
    2096       0.0010321      -0.0581235             0.0       0       0
    2097   -9.468379e-04      -0.0581464             0.0       0       0
    2098      -0.0029076      -0.0574702             0.0       0       0
    2099      -0.0045615       -0.058324             0.0       0       0
    2100      -0.0052236      -0.0510105             0.0       0       0
    2101      -0.0045995      -0.0536858             0.0       0       0
    2102      -0.0049311      -0.0547932             0.0       0       0
    2103      -0.0041943      -0.0557221             0.0       0       0
    2106         -0.0045      -0.0289324       0.0184461       0       0
    2107         -0.0045      -0.0299517       0.0167472       0       0
    2109         -0.0045      -0.0319903       0.0133494       0       0
    2113         -0.0045      -0.0360676       0.0065539       0       0
    2114         -0.0021       -0.038116       0.0054724       0       0
    2118         -0.0021      -0.0309807       0.0173646       0       0
    2120         -0.0021      -0.0330193       0.0139668       0       0
    2124         -0.0021      -0.0370966       0.0071713       0       0
    2128         -0.0021       -0.028942       0.0207624       0       0
    2130         -0.0021       -0.028024       0.0220708       0       0
    2131         -0.0021       -0.026929       0.0232352       0       0
    2135   -5.000000e-04      -0.0274712       0.0238234       0       0
    2136   -5.000000e-04      -0.0286444       0.0225758       0       0
    2139         -0.0021      -0.0239675       0.0251984       0       0
    2144   -5.000000e-04      -0.0249295       0.0256177       0       0
    2147   -5.000000e-04      -0.0223112       0.0266258       0       0
    2148   -5.000017e-04      -0.0261324       0.0248913       0       0
    2149   -5.000000e-04      -0.0236499       0.0261985       0       0
    2153    5.000000e-04      -0.0236499       0.0261985       0       0
    2154    5.000000e-04      -0.0249295       0.0256177       0       0
    2157          0.0021      -0.0239675       0.0251984       0       0
    2158    5.000000e-04      -0.0261324       0.0248913       0       0
    2160    5.000000e-04      -0.0223112       0.0266258       0       0
    2165          0.0045      -0.0432473    2.959054e-04       0       0
    2166          0.0045      -0.0382573        0.003279       0       0
    2167          0.0045       -0.039713       0.0019619       0       0
    2172           0.008      -0.0456619   -9.313226e-10       0       0
    2175           0.008       -0.043953    1.470922e-04       0       0
    2176           0.008      -0.0422944    5.840436e-04       0       0
    2177           0.008      -0.0407349        0.001298       0       0
    2178           0.008      -0.0393203        0.002268       0       0
    2179           0.008      -0.0380923       0.0034654       0       0
    2182       0.0064102       -0.043123    3.276709e-04       0       0
    2183       0.0062501      -0.0381743       0.0033716       0       0
    2184       0.0062525       -0.039519       0.0021092       0       0
    2185       0.0062701      -0.0411038       0.0010992       0       0
    2190          0.0045      -0.0350483       0.0082528       0       0
    2192          0.0045      -0.0330097       0.0116506       0       0
    2193          0.0045      -0.0319903       0.0133494       0       0
    2194          0.0045       -0.030971       0.0150483       0       0
    2198       0.0031584      -0.0281846       0.0203079       0       0
    2201          0.0021      -0.0360773       0.0088702       0       0
    2203          0.0021      -0.0340387        0.012268       0       0
    2204          0.0021      -0.0330193       0.0139668       0       0
    2205          0.0021          -0.032       0.0156657       0       0
    2206          0.0021      -0.0309807       0.0173646       0       0
    2208       0.0031584      -0.0373586        0.005018       0       0
    2211          0.0021       -0.028942       0.0207624       0       0
    2213          0.0021       -0.028024       0.0220708       0       0
    2218    5.000000e-04      -0.0274712       0.0238234       0       0
    2221       0.0432473         -0.0045    2.959007e-04       0       0
    2222       0.0444651       -0.004395    7.187016e-05       0       0
    2227        0.037087          -0.008        0.004855       0       0
    2228        0.037087        -0.00625        0.004855       0       0
    2229       0.0456619          -0.008   -9.313226e-10       0       0
    2230       0.0380923          -0.008       0.0034654       0       0
    2231       0.0393203          -0.008        0.002268       0       0
    2232       0.0407349          -0.008        0.001298       0       0
    2233       0.0422944          -0.008    5.840436e-04       0       0
    2234        0.043953          -0.008    1.470922e-04       0       0
    2238       0.0381739      -0.0062499        0.003372       0       0
    2239       0.0395188      -0.0062524       0.0021094       0       0
    2240       0.0411025      -0.0062697       0.0010999       0       0
    2241       0.0431248       -0.006411    3.271997e-04       0       0
    2243        0.037087         -0.0045        0.004855       0       0
    2244       0.0289324         -0.0045       0.0184461       0       0
    2245       0.0299517         -0.0045       0.0167472       0       0
    2246        0.030971         -0.0045       0.0150483       0       0
    2247       0.0319903         -0.0045       0.0133494       0       0
    2248       0.0330097         -0.0045       0.0116506       0       0
    2249        0.034029         -0.0045       0.0099517       0       0
    2250       0.0350483         -0.0045       0.0082528       0       0
    2251       0.0360676         -0.0045       0.0065539       0       0
    2253       0.0373586      -0.0031584        0.005018       0       0
    2268        0.028024         -0.0021       0.0220708       0       0
    2269        0.026929         -0.0021       0.0232352       0       0
    2272        0.029628   -5.000000e-04       0.0211739       0       0
    2273       0.0274712   -5.000000e-04       0.0238234       0       0
    2274       0.0286444   -5.000000e-04       0.0225758       0       0
    2287       0.0236499   -5.000000e-04       0.0261985       0       0
    2288       0.0249295   -5.000000e-04       0.0256177       0       0
    2300      -0.0450063      -0.0028655    3.314935e-04       0       0
    2306      -0.0466069   -5.000000e-04    2.834215e-04       0       0
    2308      -0.0455745    5.000000e-04       0.0010801       0       0
    2310      -0.0450063       0.0028655    3.314966e-04       0       0
    2311      -0.0478798    5.000000e-04   -5.820766e-11       0       0
    2312      -0.0476024       0.0019359   -5.820766e-11       0       0
    2313      -0.0468199       0.0031724   -5.820766e-11       0       0
    2314      -0.0466069    5.000000e-04    2.834207e-04       0       0
    2322      -0.0432473          0.0045    2.959035e-04       0       0
    2328      -0.0433238       0.0031583    6.032530e-04       0       0
    2332      -0.0444955       0.0019277       0.0012527       0       0
    2335      -0.0444651        0.004395    7.187296e-05       0       0
    2339       -0.038116          0.0021       0.0054724       0       0
    2340      -0.0299613          0.0021       0.0190635       0       0
    2344      -0.0340387          0.0021        0.012268       0       0
    2346      -0.0360773          0.0021       0.0088702       0       0
    2347      -0.0370966          0.0021       0.0071713       0       0
    2350      -0.0306473    5.000000e-04       0.0194751       0       0
    2352       -0.032686    5.000000e-04       0.0160773       0       0
    2353      -0.0337053    5.000000e-04       0.0143784       0       0
    2354      -0.0347246    5.000000e-04       0.0126796       0       0
    2357      -0.0377826    5.000000e-04       0.0075829       0       0
    2360      -0.0274712    5.000000e-04       0.0238234       0       0
    2366       0.0444955       0.0019277       0.0012527       0       0
    2371       0.0433238       0.0031583    6.032533e-04       0       0
    2372       0.0444651        0.004395    7.187296e-05       0       0
    2374       0.0432473          0.0045    2.959063e-04       0       0
    2377        0.041398          0.0045    9.546103e-04       0       0
    2379       0.0373586       0.0031584        0.005018       0       0
    2380        0.043537          0.0021       0.0014604       0       0
    2396           0.032          0.0021       0.0156657       0       0
    2397       0.0309807          0.0021       0.0173646       0       0
    2398       0.0299613          0.0021       0.0190635       0       0
    2399        0.038802    5.000000e-04        0.005884       0       0
    2409       0.0456619       0.0040693   -5.820766e-11       0       0
    2410       0.0478798    5.000000e-04   -5.820766e-11       0       0
    2411       0.0468199       0.0031724   -5.820766e-11       0       0
    2412       0.0476024       0.0019359   -5.820766e-11       0       0
    2414       0.0450063       0.0028655    3.314966e-04       0       0
    2415       0.0455745    5.000000e-04       0.0010801       0       0
    2416       0.0466069    5.000000e-04    2.834207e-04       0       0
    2422       0.0466069   -5.000000e-04    2.834215e-04       0       0
    2431       0.0261324   -5.000017e-04       0.0248913       0       0
    2443      -0.0433238      -0.0031584    6.032462e-04       0       0
    2445      -0.0444651       -0.004395    7.187115e-05       0       0
    2450      -0.0391459         -0.0021       0.0040855       0       0
    2451      -0.0404269         -0.0021       0.0029265       0       0
    2452      -0.0419097         -0.0021       0.0020401       0       0
    2459      -0.0373586      -0.0031584        0.005018       0       0
    2464      -0.0370966         -0.0021       0.0071713       0       0
    2465      -0.0360773         -0.0021       0.0088702       0       0
    2466       -0.035058         -0.0021       0.0105691       0       0
    2467      -0.0340387         -0.0021        0.012268       0       0
    2468      -0.0330193         -0.0021       0.0139668       0       0
    2469          -0.032         -0.0021       0.0156657       0       0
    2470      -0.0309807         -0.0021       0.0173646       0       0
    2477      -0.0347246   -5.000000e-04       0.0126796       0       0
    2479       -0.032686   -5.000000e-04       0.0160773       0       0
    2480      -0.0316667   -5.000000e-04       0.0177762       0       0
    2481      -0.0306473   -5.000000e-04       0.0194751       0       0
    2483         -0.0045      -0.0432473    2.959017e-04       0       0
    2484         -0.0045      -0.0382573        0.003279       0       0
    2486         -0.0045       -0.041398    9.546103e-04       0       0
    2488      -0.0031584      -0.0433238    6.032456e-04       0       0
    2492         -0.0021      -0.0419097       0.0020401       0       0
    2493      -0.0031584      -0.0373586        0.005018       0       0
    2498      -0.0028655      -0.0450063    3.314935e-04       0       0
    2501         -0.0021       -0.043537       0.0014604       0       0
    2513    5.000000e-04      -0.0478798   -2.328306e-10       0       0
    2515    5.000000e-04      -0.0466069    2.834206e-04       0       0
    2516   -5.000000e-04      -0.0455745       0.0010801       0       0
    2518      -0.0040693      -0.0456619   -6.984919e-10       0       0
    2519   -4.999997e-04      -0.0478798             0.0       0       0
    2523   -5.000000e-04      -0.0466069    2.834215e-04       0       0
    2524      -0.0019277      -0.0444955       0.0012527       0       0
    2528         -0.0021      -0.0299613       0.0190635       0       0
    2530         -0.0021          -0.032       0.0156657       0       0
    2532         -0.0021      -0.0340387        0.012268       0       0
    2533         -0.0021       -0.035058       0.0105691       0       0
    2534         -0.0021      -0.0360773       0.0088702       0       0
    2537   -5.000000e-04       -0.038802        0.005884       0       0
    2538   -5.000000e-04      -0.0306473       0.0194751       0       0
    2541   -5.000000e-04      -0.0337053       0.0143784       0       0
    2544   -5.000000e-04      -0.0367633       0.0092818       0       0
    2547   -5.000000e-04       -0.029628       0.0211739       0       0
    2551    5.000000e-04       -0.029628       0.0211739       0       0
    2552    5.000000e-04      -0.0286444       0.0225758       0       0
    2556       0.0028655      -0.0450063    3.314966e-04       0       0
    2560        0.004395      -0.0444651    7.187296e-05       0       0
    2565          0.0045       -0.041398    9.546103e-04       0       0
    2570          0.0021      -0.0404269       0.0029265       0       0
    2574       0.0031583      -0.0433238    6.032530e-04       0       0
    2579          0.0021      -0.0370966       0.0071713       0       0
    2581          0.0021       -0.035058       0.0105691       0       0
    2586          0.0021      -0.0299613       0.0190635       0       0
    2590    5.000000e-04      -0.0367633       0.0092818       0       0
    2594    5.000000e-04       -0.032686       0.0160773       0       0
    2599       0.0382573         -0.0045        0.003279       0       0
    2600        0.039713         -0.0045       0.0019619       0       0
    2601        0.041398         -0.0045    9.546103e-04       0       0
    2603       0.0433238      -0.0031584    6.032456e-04       0       0
    2613       0.0450063      -0.0028655    3.314935e-04       0       0
    2619        0.028942         -0.0021       0.0207624       0       0
    2621       0.0299613         -0.0021       0.0190635       0       0
    2622       0.0309807         -0.0021       0.0173646       0       0
    2623           0.032         -0.0021       0.0156657       0       0
    2624       0.0330193         -0.0021       0.0139668       0       0
    2625       0.0340387         -0.0021        0.012268       0       0
    2626        0.035058         -0.0021       0.0105691       0       0
    2627       0.0360773         -0.0021       0.0088702       0       0
    2628       0.0370966         -0.0021       0.0071713       0       0
    2631       0.0306473   -5.000000e-04       0.0194751       0       0
    2632       0.0316667   -5.000000e-04       0.0177762       0       0
    2633        0.032686   -5.000000e-04       0.0160773       0       0
    2634       0.0337053   -5.000000e-04       0.0143784       0       0
    2635       0.0347246   -5.000000e-04       0.0126796       0       0
    2636        0.035744   -5.000000e-04       0.0109807       0       0
    2637       0.0367633   -5.000000e-04       0.0092818       0       0
    2638       0.0377826   -5.000000e-04       0.0075829       0       0
    2639      -0.0444955      -0.0019277       0.0012527       0       0
    2647      -0.0455745   -5.000000e-04       0.0010801       0       0
    2648      -0.0437302    5.000000e-04       0.0022367       0       0
    2649      -0.0447446    5.000000e-04       0.0018055       0       0
    2650      -0.0437302   -5.000000e-04       0.0022367       0       0
    2651      -0.0447446   -5.000000e-04       0.0018055       0       0
    2654       -0.043537          0.0021       0.0014604       0       0
    2660      -0.0419097          0.0021       0.0020401       0       0
    2661      -0.0404269          0.0021       0.0029265       0       0
    2662      -0.0391459          0.0021       0.0040855       0       0
    2663       -0.038802    5.000000e-04        0.005884       0       0
    2665      -0.0422508    5.000000e-04       0.0027637       0       0
    2666      -0.0409028    5.000000e-04       0.0035695       0       0
    2667      -0.0397382    5.000000e-04       0.0046232       0       0
    2669       -0.029628   -5.000000e-04       0.0211739       0       0
    2670       -0.029628    5.000000e-04       0.0211739       0       0
    2671       -0.038802   -5.000000e-04        0.005884       0       0
    2672      -0.0377826   -5.000000e-04       0.0075829       0       0
    2673      -0.0367633   -5.000000e-04       0.0092818       0       0
    2674       -0.035744   -5.000000e-04       0.0109807       0       0
    2676      -0.0337053   -5.000000e-04       0.0143784       0       0
    2682      -0.0316667    5.000000e-04       0.0177762       0       0
    2686       -0.035744    5.000000e-04       0.0109807       0       0
    2687      -0.0367633    5.000000e-04       0.0092818       0       0
    2696        0.038116          0.0021       0.0054724       0       0
    2697       0.0419097          0.0021       0.0020401       0       0
    2698       0.0404269          0.0021       0.0029265       0       0
    2699       0.0391459          0.0021       0.0040855       0       0
    2703       0.0422508    5.000000e-04       0.0027637       0       0
    2704       0.0409028    5.000000e-04       0.0035695       0       0
    2706        0.029628    5.000000e-04       0.0211739       0       0
    2709       0.0377826    5.000000e-04       0.0075829       0       0
    2710       0.0367633    5.000000e-04       0.0092818       0       0
    2711        0.035744    5.000000e-04       0.0109807       0       0
    2712       0.0347246    5.000000e-04       0.0126796       0       0
    2713       0.0337053    5.000000e-04       0.0143784       0       0
    2714        0.032686    5.000000e-04       0.0160773       0       0
    2715       0.0316667    5.000000e-04       0.0177762       0       0
    2716       0.0306473    5.000000e-04       0.0194751       0       0
    2726       0.0455745   -5.000000e-04       0.0010801       0       0
    2729       0.0447446   -5.000000e-04       0.0018055       0       0
    2730       0.0437302    5.000000e-04       0.0022367       0       0
    2731       0.0447446    5.000000e-04       0.0018055       0       0
    2732       0.0444955      -0.0019277       0.0012527       0       0
    2740       -0.043537         -0.0021       0.0014604       0       0
    2751         -0.0021      -0.0391459       0.0040855       0       0
    2752         -0.0021      -0.0404269       0.0029265       0       0
    2758   -5.000000e-04      -0.0397382       0.0046232       0       0
    2766       0.0019277      -0.0444955       0.0012527       0       0
    2767          0.0021       -0.043537       0.0014604       0       0
    2768    5.000000e-04      -0.0455745       0.0010801       0       0
    2774   -5.000000e-04      -0.0437302       0.0022367       0       0
    2775   -5.000000e-04      -0.0447446       0.0018055       0       0
    2777    5.000000e-04      -0.0447446       0.0018055       0       0
    2781    5.000000e-04      -0.0377826       0.0075829       0       0
    2783    5.000000e-04       -0.035744       0.0109807       0       0
    2784    5.000000e-04      -0.0347246       0.0126796       0       0
    2785    5.000000e-04      -0.0337053       0.0143784       0       0
    2787    5.000000e-04      -0.0316667       0.0177762       0       0
    2788    5.000000e-04      -0.0306473       0.0194751       0       0
    2791   -5.000000e-04      -0.0316667       0.0177762       0       0
    2792   -5.000000e-04       -0.032686       0.0160773       0       0
    2794   -5.000000e-04      -0.0347246       0.0126796       0       0
    2795   -5.000000e-04       -0.035744       0.0109807       0       0
    2797   -5.000000e-04      -0.0377826       0.0075829       0       0
    2799          0.0021       -0.038116       0.0054724       0       0
    2800          0.0021      -0.0419097       0.0020401       0       0
    2802          0.0021      -0.0391459       0.0040855       0       0
    2806    5.000000e-04      -0.0422508       0.0027637       0       0
    2807    5.000000e-04      -0.0409028       0.0035695       0       0
    2808    5.000000e-04      -0.0397382       0.0046232       0       0
    2809        0.038116         -0.0021       0.0054724       0       0
    2810        0.043537         -0.0021       0.0014604       0       0
    2811       0.0391459         -0.0021       0.0040855       0       0
    2812       0.0404269         -0.0021       0.0029265       0       0
    2813       0.0419097         -0.0021       0.0020401       0       0
    2814        0.038802   -5.000000e-04        0.005884       0       0
    2815       0.0437302   -5.000000e-04       0.0022367       0       0
    2816       0.0422508   -5.000000e-04       0.0027637       0       0
    2817       0.0409028   -5.000000e-04       0.0035695       0       0
    2818       0.0397382   -5.000000e-04       0.0046232       0       0
    2826      -0.0422508   -5.000000e-04       0.0027637       0       0
    2827      -0.0409028   -5.000000e-04       0.0035695       0       0
    2828      -0.0397382   -5.000000e-04       0.0046232       0       0
    2838       0.0397382    5.000000e-04       0.0046232       0       0
    2841   -5.000000e-04      -0.0422508       0.0027637       0       0
    2842   -5.000000e-04      -0.0409028       0.0035695       0       0
    2844    5.000000e-04      -0.0437302       0.0022367       0       0
    2845    5.000000e-04       -0.038802        0.005884       0       0
*END
