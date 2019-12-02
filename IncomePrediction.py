import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from numpy import sqrt
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
##import catboost as cat
from datetime import datetime
import lightgbm as lgb


###############Copied Function and slightly modified for Target Encoding to use on Professions Feature set#####################
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Return smooth mean distribution
    return smooth


def main():

    ###############GENERAL APPROACH#######################
    
    1. Fill in NA numeric variables with median
    2. Fill in NA categorical variables with 'Unknown' feature
    3. Remove Targets considered 3 standard deviations from the mean as outliers
    4. Mean Normalize all numeric variables
    5. One hot encoding on all categorical variables except Professions and Countries
    6. Target Encoding on Professions and Countries
    7. Dropped Glasses variable from Data Set.
    8. Trained LightGBM and output predictions on testData.


    #######################DATA INPUT#########################################

    training = pd.read_csv('tcd-ml-1920-group-income-train.csv')
    test = pd.read_csv('tcd-ml-1920-group-income-test.csv')

    #####################DATA CLEANING#########################################

    ##Fix easy NA fills
    training['Gender'] = training['Gender'].fillna('UnknownGender')
    training['Hair Color'] = training['Hair Color'].fillna('UnknownHair')
    training['Satisfation with employer'] = training['Satisfation with employer'].fillna('UnknownSatisfaction')
    training['Wears Glasses'] = training['Wears Glasses'].fillna('0')
    training['University Degree'] = training['University Degree'].fillna('UnknownDegree')
    training['Country'] = training['Country'].fillna('UnknownCountry')
    training['Profession'] = training['Profession'].fillna('OtherProfession')
    training['Year of Record'] = training['Year of Record'].fillna(training['Year of Record'].median())
    training['Age'] = training['Age'].fillna(training['Age'].median())
    training['Size of City'] = training['Size of City'].fillna(training['Size of City'].median())
    training['Body Height [cm]'] = training['Body Height [cm]'].fillna(training['Body Height [cm]'].median())


    ##Drop all NAs
    training = training.dropna()

    ## Remove outliers 3 Std. Deviations away from mean
    incomeStd = training['Total Yearly Income [EUR]'].std()
    incomeMean = training['Total Yearly Income [EUR]'].mean()
    training = training[training['Total Yearly Income [EUR]'] < (incomeMean+3*incomeStd)]
    target = training['Total Yearly Income [EUR]']
    target = target._set_name('Total Yearly Income')

    ##Init some zero arrays size of Training and Test
    rows = training.shape[0]
    testRows = test.shape[0]

    ########################DATA PROCESSING###############################

    ##Income Added to Salary
    incomeAdded = training['Yearly Income in addition to Salary (e.g. Rental Income)']
    incomeAdded = incomeAdded.str.replace('EUR', '')
    incomeAdded = incomeAdded.astype('float64')
    testIncomeAdded = test['Yearly Income in addition to Salary (e.g. Rental Income)']
    testIncomeAdded = testIncomeAdded.str.replace('EUR', '')
    testIncomeAdded = testIncomeAdded.astype('float64')
    incomeAdded = incomeAdded._set_name('Yearly Income in addition to Salary')
    testIncomeAdded = testIncomeAdded._set_name('Yearly Income in addition to Salary')

    ##AdjustedTarget
    ##adjustedTarget = incomeAdded._set_name('Total Yearly Income')
    ##adjustedTarget = target - adjustedTarget
    ##adjustedTargetDataframe = pd.DataFrame(adjustedTarget)

    ##pd.DataFrame(adjustedTarget).to_csv('adjusted-Dataset.csv', index_label='Instance')


    ##Year of Record
    yearOfRecord = training['Year of Record']
    testYearOfRecord = test['Year of Record']
    a = testYearOfRecord.unique()

    testYearOfRecord = testYearOfRecord.replace(a[60], yearOfRecord.mean())
    testYearOfRecord = (testYearOfRecord - yearOfRecord.mean()) / yearOfRecord.std()
    yearOfRecord = (yearOfRecord-yearOfRecord.mean())/yearOfRecord.std()


    ## Crime Level in the City of Employment
    crimeLevel = training['Crime Level in the City of Employement']
    testCrimeLevel = test['Crime Level in the City of Employement']
    testCrimeLevel = (testCrimeLevel-crimeLevel.mean())/crimeLevel.std()
    crimeLevel = (crimeLevel-crimeLevel.mean())/crimeLevel.std()


    ##Gender
    gender = training['Gender']
    gender = gender.replace(['unknown','0'],'UnknownGender')
    gender = gender.replace(['f'],'UnknownGender')
    gender = pd.get_dummies(gender)

    ##Target Encoding for Gender, didn't improve score.
    ##genderTarget = pd.concat([gender, target], axis=1)
    ##genderTargetEncoded = calc_smooth_mean(genderTarget,'Gender','Total Yearly Income',50)
    ##gender = gender.map(genderTargetEncoded)

    testGender = test['Gender']
    testGender = testGender.fillna('UnknownGender')
    testGender = testGender.replace(['unknown','0'],'UnknownGender')
    testGender = testGender.replace(['f'],'UnknownGender')
    testGender = pd.get_dummies(testGender)

    ##testGender = testGender.map(genderTargetEncoded)


    ##Work Experience in Current Job [years]
    workExperience = training['Work Experience in Current Job [years]']
    testWorkExperience = test['Work Experience in Current Job [years]']
    workExperience = workExperience.astype('float64')
    testWorkExperience = testWorkExperience.replace(['#NUM!'], workExperience.mean())
    testWorkExperience = testWorkExperience.astype('float64')
    testWorkExperience = (testWorkExperience-workExperience.mean())/workExperience.std()
    workExperience = (workExperience - workExperience.mean()) / workExperience.std()
    workExperience = workExperience._set_name('Work Experience in Current Job')
    testWorkExperience = testWorkExperience._set_name('Work Experience in Current Job')


    ##Housing Situation
    housingSit = training['Housing Situation']
    housingSit = housingSit.replace(['0',0],'Homeless')
    housingSit = housingSit.replace(['nA'],'UnknownHousing')
    housingSit = pd.get_dummies(housingSit)

    ##Target Encoding for Housing Situation, poor results, used to study data.
    ##housingSitandTarget = pd.concat([housingSit, target], axis=1)
    ##targetEncodedHousingSit = calc_smooth_mean(housingSitandTarget, 'Housing Situation', 'Total Yearly Income', 50)
    ##housingSit = housingSit.map(targetEncodedHousingSit)
    

    testHousingSit = test['Housing Situation']
    testHousingSit = testHousingSit.replace(['0'],'Homeless')
    testHousingSit = testHousingSit.replace(['nA'],'UnknownHousing')
    testHousingSit = pd.get_dummies(testHousingSit)

    ##testHousingSit = testHousingSit.map(targetEncodedHousingSit)


    ##Satisfaction with employer
    satisfaction = training['Satisfation with employer']
    satisfaction = pd.get_dummies(satisfaction)

    ##satisfactionAndTarget = pd.concat([satisfaction, target], axis=1)
    ##satisfactionTargetEncoded = calc_smooth_mean(satisfactionAndTarget,'Satisfation with employer','Total Yearly Income',50)
    ##satisfaction = satisfaction.map(satisfactionTargetEncoded)

    testSatisfaction = test['Satisfation with employer']
    testSatisfaction = testSatisfaction.fillna('UnknownSatisfaction')
    ##testSatisfaction = testSatisfaction.map(satisfactionTargetEncoded)
    testSatisfaction = pd.get_dummies(testSatisfaction)


    ##Age
    age = training['Age']
    testAge = test['Age']
    testAge = (testAge-age.mean())/age.std()
    age = (age-age.mean())/age.std()


    ##CitySize
    citySize = training['Size of City']
    testCitySize = test['Size of City']
    testCitySize = (testCitySize-citySize.mean())/citySize.std()
    citySize = (citySize-citySize.mean())/citySize.std()


    ##Height
    height = training['Body Height [cm]']
    testHeight = test['Body Height [cm]']
    testHeight = (testHeight-height.mean())/height.std()
    height = (height-height.mean())/height.std()
    height = height._set_name('Body Height')
    testHeight = testHeight._set_name('Body Height')


    ##Glasses is completely useless, 50/50 split in both datasets and mean of Glasses and non Glasses are pretty similar. SACK
    ##glasses = training['Wears Glasses']
    ##glassesAndTraining = pd.concat([glasses,target],axis=1)
    ##targetEncodedGlasses = calc_smooth_mean(glassesAndTraining, 'Wears Glasses', 'Total Yearly Income',0)
    ##glasses = pd.get_dummies(glasses)
    ##glasses= glasses.drop(glasses.columns[0], axis=1)
    ##testGlasses = test['Wears Glasses']
    ##testGlasses = pd.get_dummies(testGlasses)
    ##testGlasses = testGlasses.drop(testGlasses.columns[0],axis=1)


    ##Uni-Degree
    uniDeg = training['University Degree']
    uniDeg = uniDeg.replace(['0'],'MysteryDegree')
    ##uniDegTarget = pd.concat([uniDeg, target], axis=1)
    ##uniDegTargetEncoded = calc_smooth_mean(uniDegTarget, 'University Degree', 'Total Yearly Income', 0)
    ##uniDeg = uniDeg.map(uniDegTargetEncoded)
    uniDeg = pd.get_dummies(uniDeg)

    testUniDeg = test['University Degree']
    testUniDeg = testUniDeg.fillna('UnknownDegree')
    testUniDeg = testUniDeg.replace(['0'],'MysteryDegree')
    ##testUniDeg = testUniDeg.map(uniDegTargetEncoded)
    testUniDeg = pd.get_dummies(testUniDeg)


    ##Profession
    profession = training['Profession']
    testProfession = test['Profession']
    testProfession = testProfession.fillna('OtherProfession')
    professionAndTarget = pd.concat([profession,target], axis=1)
    targetEncodedProfessions = calc_smooth_mean(professionAndTarget,'Profession','Total Yearly Income',50)
    profession=profession.map(targetEncodedProfessions)
    testProfession = testProfession.map(targetEncodedProfessions)


    ##Country
    country = training['Country']
    testCountry = test['Country']
    # remove Nan
    country = country.replace(['0'], 'OtherCountry')
    # Set Threshold for 'OtherCountry' to 50 in training data (variable name is arbitrary, other is used for Gender)
    ##countryCounts = country.value_counts()
    ##indexA = countryCounts[countryCounts <= 50].index
    ##country = country.replace(indexA, "OtherCountry")
    ##testCountry = testCountry.replace(indexA, "OtherCountry")

    # remove nan from test -> OtherCountry
    listA = country.unique()
    listB = testCountry.unique()
    testCountry = testCountry.replace(listB[168], 'OtherCountry')
    listB = testCountry.unique()
    # Set countries only occuring in test data -> OtherCountry
    listDiffB = np.setdiff1d(listB, listA)
    testCountry = testCountry.replace(listDiffB, 'OtherCountry')
    listDiffA = np.setdiff1d(listA,listB)
    country = country.replace(listDiffA, 'OtherCountry')

    countryAndTarget = pd.concat([country,target],axis=1)
    countryTargetEncoded = calc_smooth_mean(countryAndTarget, 'Country','Total Yearly Income',100)
    country=country.map(countryTargetEncoded)
    testCountry=testCountry.map(countryTargetEncoded)


    ##Hair Colour
    hairCol = training['Hair Color']
    testHairCol = test['Hair Color']

    ##hairAndTarget = pd.concat([hairCol,target],axis=1)
    ##hairTargetEncoded = calc_smooth_mean(hairAndTarget,'Hair Color','Total Yearly Income',0)

    hairCol = hairCol.replace(['0'],'MysteryHair')
    testHairCol = testHairCol.replace(['0'],'MysteryHair')
    hairCol = hairCol.replace(['Unknown'], 'UnknownHair')
    testHairCol = testHairCol.replace(['Unknown'], 'UnknownHair')


    hairCol = pd.get_dummies(hairCol)
    testHairCol = pd.get_dummies(testHairCol)


    ##Join Data

    modelData = pd.concat([yearOfRecord,crimeLevel,workExperience,age,citySize, height, profession, hairCol,
                           gender,housingSit,satisfaction,uniDeg,country,incomeAdded], axis=1)

    testData = pd.concat([testYearOfRecord,testCrimeLevel,testWorkExperience,testAge,testCitySize,
                          testHeight, testProfession, testHairCol,
                          testGender,testHousingSit,testSatisfaction,testUniDeg
                          ,testCountry,testIncomeAdded], axis=1)

    ##Simple method to export Datasets to team mates for them to play with various parameters and algos.
    ##pd.DataFrame(modelData).to_csv('modelDataFile.csv', index_label='Instance')
    ##pd.DataFrame(testData).to_csv('testDataFile.csv', index_label='Instance')


    ############################MODEL PREDICTIONS#######################################

    ##bestSub = pd.read_csv('best-submission-so-far.csv')
    ##currentSub = pd.read_csv('tcd-ml-1920-group-income-submission-catBoost.csv')


    ##MAE = mean_absolute_error(bestSub, currentSub)
    ##print('MAE between XGB')
    ##print(MAE)




    ###########################XGBOOST LETS GOOOOOO####################################

    ##training_dMatrix = xgb.DMatrix(modelData,target)
    ##training_dMatrix.save_binary('training_buffer')
    ##test_Dmatric = xgb.DMatrix(testData)
    ##test_Dmatric.save_binary('test buffer')

    ##dMatrix = xgb.DMatrix('training_buffer')
    ##tMatrix= xgb.DMatrix('test buffer')
    ##param = {'max_depth':6, 'eta':0.1, 'eval_metric': 'mae'}

    ##bst1 = xgb.train(param, dMatrix, 1000)
    ##plot_importance(bst1, max_num_features=50)
    ##plt.show()

    ##ypred = bst1.predict(tMatrix)

    ########################CATBOOST LETS GOOOOOOOOOOOOOO#################################



    ########################LGB LETS GOOOOOOOOOOOOOO#################################

    # import lightgbm as lgb

    # print('Creating lgb dataset')
    # lgb_train = lgb.Dataset(modelData.values, label= target.values)
    # params = {
    # 'task': 'train'
    # , 'boosting_type': 'gbdt'
    # , 'objective': 'regression'
    # , 'metric': 'mae'
    # , 'min_data': 1
    # , 'verbose': -1
    # }

    # gbm = lgb.train(params, lgb_train, num_boost_round=40000)
    # lgb_pred = gbm.predict(testData.values)

    # # Get the mean absolute error on the validation data :

    # # MAE = mean_absolute_error(Y_test , lgb_pred)
    # # print('lgb validation MAE = ',MAE)

    # print('Writing lgb')
    # with open('lgbPred.csv', 'w') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerows(map(lambda x: [x], lgb_pred))
    # csvFile.close()

    #######################lgb LOCAL LETS GOOOOOOOOOOOOOO#################################

    import lightgbm as lgb

    print('Creating lgb dataset')
    lgb_train = lgb.Dataset(modelData.values, label=target.values)
    params = {
        'task': 'train'
        , 'boosting_type': 'gbdt'
        , 'objective': 'regression'
        , 'metric': 'mae'
        , 'min_data': 1
        , 'verbose': -1
    }

    gbm = lgb.train(params, lgb_train, num_boost_round=50000)
    ypred = gbm.predict(testData)

    # Get the mean absolute error on the validation data :
    ##MAE = mean_absolute_error(Y_test, lgb_pred)
    ##print('lgb validation MAE = ', MAE)

    ##ypred = ypred + testIncomeAdded

    ########################PRINT YPRED TO CSV FILE#####################################

    pd.DataFrame(ypred).to_csv('tcd-ml-1920-group-income-submission.csv', index_label='Instance', header=['Total Yearly Income [EUR]'])
    print('Complete')
    print(datetime.now())

main()



