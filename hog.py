from rpsimgproc import *

generateHOGFeatures('data_bin.csv', binary=True)
generateHOGFeatures('data.csv', binary=False)
