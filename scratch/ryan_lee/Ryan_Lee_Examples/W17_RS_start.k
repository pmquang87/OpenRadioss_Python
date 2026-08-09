$# LS-DYNA Keyword file created by LS-PrePost(R) V4.5.1 - 14Oct2017
$# Created on Jul-07-2018 (10:03:22)
$ Created by BW Lee, Copyright@BW Engineering 2018
$ Description
$===============================================================================
$ Restart analysis file for Chapter 17. Floor panel load analysis
$ Implicit analysis - Modification of loading_x, loading_y
$ Unit system : m, kg, sec, Pa
$===============================================================================
*KEYWORD
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol     
       2.0         0       0.0       0.01.000000E8         0
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid      
      0.05         0         0         0         0
$#   ioopt     
         0
*DEFINE_CURVE_TITLE
Load_x
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint
         2         0       1.0       1.0       0.0       0.0         0         0
$#                a1                  o1  
                 0.0                 0.0
                 1.0                 0.0
                 2.1              1600.0
*DEFINE_CURVE_TITLE
Load_y
$#    lcid      sidr       sfa       sfo      offa      offo    dattyp     lcint
         3         0       1.0       1.0       0.0       0.0         0         0
$#                a1                  o1  
                 0.0                 0.0
                 1.0                 0.0
                 2.1              1200.0
*CHANGE_CURVE_DEFINITION
$#    lcid    
         2
*CHANGE_CURVE_DEFINITION
$#    lcid    
         3
*END
