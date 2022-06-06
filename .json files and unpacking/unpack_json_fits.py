# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:53:09 2022

@author: Tania Varesano
"""

def unpack_json_fits(file82, file86, file90) :
    import pickle
    with open(file82, 'rb') as f:
      multifit82adu = pickle.load(f)
    ampsOIII82, ampsMgIX82 = multifit82adu[0][0]
    errOIII82,  errMgIX82 = multifit82adu[0][1]
    cenOIII82, cenMgIX82  = multifit82adu[0][2]
    ampsSIV82, errSIV82, cenSIV82 = multifit82adu[1][0][0], multifit82adu[1][1][0], multifit82adu[1][2][0]
    ampsNIV82, errNIV82, cenNIV82 = multifit82adu[2][0][0], multifit82adu[2][1][0], multifit82adu[2][2][0]
    ampsNe82, errNe82, cenNe82 = multifit82adu[3][0][0], multifit82adu[3][1][0], multifit82adu[3][2][0]
    ampsSV82, ampsOIV82 = multifit82adu[4][0]
    errSV82,  errOIV82 = multifit82adu[4][1]
    cenSV82, cenOIV82 = multifit82adu[4][2]
    ampsNa82, ampsNIII82 = multifit82adu[5][0]
    errNa82,  errNIII82 = multifit82adu[5][1]
    cenNa82, cenNIII82 = multifit82adu[5][2]
    ampsOVI82, errOVI82, cenOVI82 = multifit82adu[6][0][0], multifit82adu[6][1][0], multifit82adu[6][2][0]
    
    fitamps82 = [ampsOIII82, ampsMgIX82, ampsSIV82, ampsNIV82, ampsNe82, ampsSV82, ampsOIV82, ampsNa82, ampsNIII82, ampsOVI82]
    fiterr82 = [errOIII82, errMgIX82, errSIV82, errNIV82, errNe82, errSV82, errOIV82, errNa82, errNIII82, errOVI82]
    fitcen82 = [cenOIII82, cenMgIX82, cenSIV82, cenNIV82, cenNe82, cenSV82, cenOIV82, cenNa82, cenNIII82, cenOVI82]
      
    with open(file86, 'rb') as f:
      multifit86adu = pickle.load(f)
    ampsOIII86, ampsMgIX86 = multifit86adu[0][0]
    errOIII86,  errMgIX86 = multifit86adu[0][1]
    cenOIII86, cenMgIX86  = multifit86adu[0][2]
    ampsSIV86, errSIV86, cenSIV86 = multifit86adu[1][0][0], multifit86adu[1][1][0], multifit86adu[1][2][0]
    ampsNIV86, errNIV86, cenNIV86 = multifit86adu[2][0][0], multifit86adu[2][1][0], multifit86adu[2][2][0]
    ampsNe86, errNe86, cenNe86 = multifit86adu[3][0][0], multifit86adu[3][1][0], multifit86adu[3][2][0]
    ampsSV86, ampsOIV86 = multifit86adu[4][0]
    errSV86,  errOIV86 = multifit86adu[4][1]
    cenSV86, cenOIV86 = multifit86adu[4][2]
    ampsNa86, ampsNIII86 = multifit86adu[5][0]
    errNa86,  errNIII86 = multifit86adu[5][1]
    cenNa86, cenNIII86 = multifit86adu[5][2]
    ampsOVI86, errOVI86, cenOVI86 = multifit86adu[6][0][0], multifit86adu[6][1][0], multifit86adu[6][2][0]
    
    fitamps86 = [ampsOIII86, ampsMgIX86, ampsSIV86, ampsNIV86, ampsNe86, ampsSV86, ampsOIV86, ampsNa86, ampsNIII86, ampsOVI86]
    fiterr86 = [errOIII86, errMgIX86, errSIV86, errNIV86, errNe86, errSV86, errOIV86, errNa86, errNIII86, errOVI86]
    fitcen86 = [cenOIII86, cenMgIX86, cenSIV86, cenNIV86, cenNe86, cenSV86, cenOIV86, cenNa86, cenNIII86, cenOVI86]
      
    with open(file90, 'rb') as f:
      multifit90adu = pickle.load(f)
    ampsOIII90, ampsMgIX90 = multifit90adu[0][0]
    errOIII90,  errMgIX90 = multifit90adu[0][1]
    cenOIII90, cenMgIX90  = multifit90adu[0][2]
    ampsSIV90, errSIV90, cenSIV90 = multifit90adu[1][0][0], multifit90adu[1][1][0], multifit90adu[1][2][0]
    ampsNIV90, errNIV90, cenNIV90 = multifit90adu[2][0][0], multifit90adu[2][1][0], multifit90adu[2][2][0]
    ampsNe90, errNe90, cenNe90 = multifit90adu[3][0][0], multifit90adu[3][1][0], multifit90adu[3][2][0]
    ampsSV90, ampsOIV90 = multifit90adu[4][0]
    errSV90,  errOIV90 = multifit90adu[4][1]
    cenSV90, cenOIV90 = multifit90adu[4][2]
    ampsNa90, ampsNIII90 = multifit90adu[5][0]
    errNa90,  errNIII90 = multifit90adu[5][1]
    cenNa90, cenNIII90 = multifit90adu[5][2]
    ampsOVI90, errOVI90, cenOVI90 = multifit90adu[6][0][0], multifit90adu[6][1][0], multifit90adu[6][2][0]
    
    fitamps90 = [ampsOIII90, ampsMgIX90, ampsSIV90, ampsNIV90, ampsNe90, ampsSV90, ampsOIV90, ampsNa90, ampsNIII90, ampsOVI90]
    fiterr90 = [errOIII90, errMgIX90, errSIV90, errNIV90, errNe90, errSV90, errOIV90, errNa90, errNIII90, errOVI90]
    fitcen90 = [cenOIII90, cenMgIX90, cenSIV90, cenNIV90, cenNe90, cenSV90, cenOIV90, cenNa90, cenNIII90, cenOVI90]
    
    return fitamps82, fiterr82, fitcen82, fitamps86, fiterr86, fitcen86, fitamps90, fiterr90, fitcen90