#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from helpers.data_prep import *
from helpers.eda import *
from sklearn.preprocessing import StandardScaler

import pickle

from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore")


# In[3]:


pd.set_option("display.max_columns" ,None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format" , lambda x : "%.3f" % x)


# In[4]:


path = "/Users/gokhanersoz/Desktop/VBO_Dataset/titanic.csv"
titanic = pd.read_csv(path)


# In[5]:


df = titanic.copy()


# In[6]:


def boxplot(dataframe, num_cols):
    
    i = 1
    size = 15
    num_ = len(num_cols)
    plt.figure(figsize =(12,7))
    
    for num in num_cols:
        
        plt.subplot(num_,1,i)
        sns.boxplot(dataframe[num])
        plt.xlabel(f"{num.upper()}", fontsize = size)
        plt.ylabel("Values".upper(), fontsize = size)
        plt.title(f"{num.upper()} Outliers", fontsize = size)       
        plt.tight_layout()
        i+=1
               
    plt.show()


# In[7]:


check_df(df)


# In[8]:


cat_cols ,num_cols, cat_but_car = grab_col_names(df,details = True)


# In[9]:


for cat in cat_cols:
    target_summary_with_cat(df, "Survived", cat)


# In[10]:


for num in num_cols:
    target_summary_with_num(df , "Survived", num)


# In[11]:


for cat in cat_cols:
    cat_summary(df , cat, plot = True)


# In[12]:


for num in num_cols:
    num_summary(df, num, plot = True)


# In[13]:


high_correlated_cols(df , plot = True)


# In[14]:


for num in num_cols:
    print(f"For {num.upper()} Outliers : ", check_outliers(df, num))


# In[15]:


grap_outliers(df,"Age")


# In[16]:


grap_outliers(df,"Fare")


# In[17]:


grap_outliers(df,"PassengerId")


# In[18]:


na_columns = missing_values_table(df , na_name=True)


# In[19]:


missing_vs_target(df , "Survived", na_columns = na_columns)


# In[20]:


rare_analyser(df , "Survived", cat_cols)


# In[21]:


print("Cat But Car : ", cat_but_car)
print("Num Cols : ", num_cols)
print("Cat Cols : ", cat_cols)


# In[22]:


def titanic_data_prep(dataframe):
    
    # 1 - Feature Engineering
    
    dataframe.columns = [col.upper() for col in dataframe.columns]
    
    # New Cabin Bool
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype(int)
    
    # Name Count
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
    
    # Name Word Count
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda name : len(str(name).split(" ")))
    
    # Name Dr
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda name : len([word for word in name.split(" ")                                                                       if word.startswith("Dr.")]))
    
    #dataframe["NEW_NAME_MR"] = dataframe["NAME"].apply(lambda name : len([word for word in name.split(" ") \
    #                                                                  if word.startswith("Mr.")]))
    
    #dataframe["NEW_NAME_MR"] = dataframe["NAME"].apply(lambda name : len([word for word in name.split(" ") \
    #                                                                  if word.startswith("Mr.")]))
    
    #dataframe["NEW_NAME_MRS"] = dataframe["NAME"].apply(lambda name : len([word for word in name.split(" ") \
    #                                                                  if word.startswith("Mrs.")]))
    
    #dataframe["NEW_NAME_MİSS"] = dataframe["NAME"].apply(lambda name : len([word for word in name.split(" ") \
    #                                                                  if word.startswith("Miss.")]))
    
    dataframe["NEW_TITLE"] = dataframe["NAME"].str.extract( " ([A-Za-z]+)\.", expand = False)
    
    
    # We include ourselves
    # Family Size
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    
    # New Age PClass
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    
    # Is Alone
    dataframe.loc[ (dataframe["SIBSP"] + dataframe["PARCH"] ) > 0, "NEW_IS_ALONE"] = "NO"
    dataframe.loc[ (dataframe["SIBSP"] + dataframe["PARCH"] ) == 0,"NEW_IS_ALONE"] = "YES"
    
    # Age_Cat
    # Age Level
    dataframe.loc[ (dataframe["AGE"] < 18) , "NEW_AGE_CAT"] = "young"
    dataframe.loc[ ((dataframe["AGE"] >= 18) & (dataframe["AGE"] < 56)), "NEW_AGE_CAT"] = "mature"
    dataframe.loc[ (dataframe["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"
    
    
    # Sex X Age
    # Sex - Male
    dataframe.loc[ ((dataframe["SEX"] == "male") & (dataframe["AGE"] <= 21)) ,"NEW_SEX_CAT"] = "youngmale"
    dataframe.loc[ ((dataframe["SEX"] == "male") & ((dataframe["AGE"] > 21) & (dataframe["AGE"] < 50))) ,                                                                               "NEW_SEX_CAT"] = "maturemale"
    dataframe.loc[ ((dataframe["SEX"] == "male") & (dataframe["AGE"] >= 50)) ,"NEW_SEX_CAT"] = "seniormale"
    
    # Sex - Female
    dataframe.loc[ ((dataframe["SEX"] == "female") & (dataframe["AGE"] <= 21)) ,"NEW_SEX_CAT"] = "youngfemale"
    dataframe.loc[ ((dataframe["SEX"] == "female") & ((dataframe["AGE"] > 21) & (dataframe["AGE"] < 50))) ,                                                                               "NEW_SEX_CAT"] = "maturefemale"
    dataframe.loc[ ((dataframe["SEX"] == "female") & (dataframe["AGE"] >= 50)) ,"NEW_SEX_CAT"] = "seniorfemale"
    
    
    # 2- We caught the variables!!!
    
    cat_cols , num_cols , cat_but_car = grab_col_names(dataframe, details = True)
    
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]
    
    # We destroyed outliers !!!!
    
    for col in num_cols:
        replace_with_thresholds(dataframe,col)
        
    #dataframe.drop("CABIN", inplace = True, axis = 1)
    
    # Remove unnecessary columns
    
    remove_cols = ["TICKET","NAME","CABIN"]
    dataframe.drop(remove_cols , inplace = True , axis = 1)
    
    # We Filled In The Missing Values
    
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))
    
    #missing_values_table(dataframe)
    
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    
    # Age_Level
    # Age_Cat
    
    dataframe.loc[ (dataframe["AGE"] < 18) , "NEW_AGE_CAT"] = "young"
    dataframe.loc[ ((dataframe["AGE"] >= 18) & (dataframe["AGE"] < 56)), "NEW_AGE_CAT"] = "mature"
    dataframe.loc[ (dataframe["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"
    
    # Sex X Age
    # Sex - Male
    dataframe.loc[ ((dataframe["SEX"] == "male") & (dataframe["AGE"] <= 21)) ,"NEW_SEX_CAT"] = "youngmale"
    dataframe.loc[ ((dataframe["SEX"] == "male") & ((dataframe["AGE"] > 21) & (dataframe["AGE"] < 50))) ,                                                                               "NEW_SEX_CAT"] = "maturemale"
    dataframe.loc[ ((dataframe["SEX"] == "male") & (dataframe["AGE"] >= 50)) ,"NEW_SEX_CAT"] = "seniormale"
    
    # Sex - Female
    dataframe.loc[ ((dataframe["SEX"] == "female") & (dataframe["AGE"] <= 21)) ,"NEW_SEX_CAT"] = "youngfemale"
    dataframe.loc[ ((dataframe["SEX"] == "female") & ((dataframe["AGE"] > 21) & (dataframe["AGE"] < 50))) ,                                                                               "NEW_SEX_CAT"] = "maturefemale"
    dataframe.loc[ ((dataframe["SEX"] == "female") & (dataframe["AGE"] >= 50)) ,"NEW_SEX_CAT"] = "seniorfemale"
    
    # We Filled In The Missing Values
    dataframe["EMBARKED"] = dataframe["EMBARKED"].fillna(dataframe["EMBARKED"].mode()[0])
    
    cat_cols ,num_cols, cat_but_car = grab_col_names(dataframe)
    
    for num in num_cols:
        replace_with_thresholds(dataframe, num)
    
    # 4- Label Encoding Application
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [np.int64 ,np.float64]
                   and dataframe[col].nunique() == 2 ]
    
    #print(binary_cols)
    for binary in binary_cols:
        dataframe = label_encoder(dataframe, binary)
      
    # 5- Rare Encoding Application
    dataframe = rare_encoder(dataframe,0.01)
    
    # 6- One-Hot Encoding Application
    
    #ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    
    cat_cols , num_cols, cat_but_car = grab_col_names(dataframe)
    
    
    cat_cols = [col for col in cat_cols if col not in "SURVIVED"]
        
    dataframe = one_hot_encoder(dataframe, cat_cols)
    
    #7- The process of deleting useless columns
    
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]
    
    useless_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and
                    (dataframe[col].value_counts() / len(dataframe) < 0.01).any(axis = None)]
    
    #print(useless_cols)
    
    dataframe.drop(useless_cols, axis = 1, inplace = True)
    
    dataframe.drop("PASSENGERID",inplace = True , axis = 1)
    
    #8- Standart Scaler
    
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    
    return dataframe ,scaler


# In[23]:


titanic, scaler = titanic_data_prep(df)


# In[24]:


titanic.head()


# In[25]:


import pickle

pd.to_pickle(titanic, open("Titanic_Prep.pkl", "wb"))
pd.to_pickle(scaler, open("Scaler_Titanic.pkl", "wb"))


# ## Model

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate,GridSearchCV


# In[27]:


X = titanic.drop("SURVIVED",axis = 1)
y = titanic["SURVIVED"]

print("X DataFrame Shape : {}".format(X.shape))
print("Y DataFrame Shape : {}".format(y.shape))


# # Success Evaluation (Validation) with Holdout Method

# In[28]:


from sklearn.metrics import r2_score,recall_score,precision_score,f1_score,accuracy_score,roc_auc_score,                            classification_report,confusion_matrix,roc_curve

from sklearn.model_selection import train_test_split

def score_test(model , X , y,roc_auc_plot = True, matrix = True):
    
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size =.2, random_state = 18)
    
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    print("".center(50,"#"),end = "\n\n")
    print(f" {type(model).__name__.upper()} ".center(50,"#"),end = "\n\n")
    print(" Classification Report ".center(50,"#") ,end = "\n\n")
    print(classification_report(y_test, y_pred_test),end = "\n\n")
    
    print(" Confusion Matrix ".center(50,"#") ,end = "\n\n")
    index = [["actual","actual"],["Died","Alive"]]
    columns = [["Predicted","Predicted"], ["Died","Alive"]]
    data = confusion_matrix(y_test,y_pred_test)
    last_data = pd.DataFrame(data = data , columns = columns, index = index)
    print(last_data ,end = "\n\n")
    
    
    print(" test scores ".upper().center(50,"#"),end = "\n\n")
    print(" accuracy score : ".upper(), accuracy_score(y_test,y_pred_test) ,end = "\n\n")
    print(" roc auc score : ".upper(), roc_auc_score(y_test,y_pred_test) ,end = "\n\n")
    print(" r2 score : ".upper(), r2_score(y_test,y_pred_test) ,end = "\n\n")
    print(" recall score : ".upper(), recall_score(y_test,y_pred_test) ,end = "\n\n")
    print(" f1 score : ".upper(), f1_score(y_test,y_pred_test) ,end = "\n\n")
    print(" precision score : ".upper(), precision_score(y_test,y_pred_test) ,end = "\n\n")
    
    print("".center(50,"#"),end = "\n\n")
    
    if roc_auc_plot:
        
        size = 15
        roc_score = roc_auc_score(y_test, y_prob_test)
        
        plt.figure(figsize = (7,5))
        
        fpr,tpr,threshold = roc_curve(y_test, y_prob_test)
        plt.plot(fpr,tpr , label = f"ROC AUC{round(roc_score,4)}")
        plt.plot([0,1],[0,1],"--r")
        plt.xlabel(" False Positive Rate ", fontsize = size)
        plt.ylabel(" True Positive Rate ", fontsize = size)
        plt.title(" AUC ( Area : %.3f) " % roc_score , fontsize = size)
        plt.legend()
        plt.show();
        
    if matrix:
        
        fig,ax = plt.subplots(figsize = (7,5))
        
        cm = confusion_matrix(y_test , y_pred_test)
        
        ax = sns.heatmap(data = cm, annot = True, annot_kws= {"size" : 25}, fmt = ".4g", ax = ax,
                         cmap = "rainbow", linewidths= 3, linecolor= "white", cbar = False, center = 0)
        
        size = 15
        plt.xlabel(" Predicted Label ", fontsize = size)
        plt.ylabel(" True Label ", fontsize = size)
        plt.title(f" Confusion Matrix For {type(model).__name__.upper()} ", fontsize = size)
        plt.show()


# In[29]:


score_test(LogisticRegression(),X,y)


# ## Success Evaluation with CV and Hyperparameter Optimization with GridSearchCV

# In[30]:


logistic_param_grid =    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 5),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000, 2500, 5000]
    }

models = [ ("LR" , LogisticRegression(), logistic_param_grid)]


# In[31]:


def base_model_regressor(X , y , models ,cv = 5):
    
    dictinonary = {}
    data = pd.DataFrame()
    index = 0
    
    for name,model,params in models:
        
            results = cross_validate(estimator=model,
                                     X = X,
                                     y = y,
                                     cv = cv,
                                     scoring = ["roc_auc","accuracy"])
            
            data.loc[index,"NAME"] = name
            data.loc[index,"ROC_AUC"] = results["test_roc_auc"].mean()
            data.loc[index,"ACCURACY"] = results["test_accuracy"].mean()
            data.loc[index,"FIT_TIME"] = results["fit_time"].mean()
            data.loc[index,"SCORE_TIME"] = results["score_time"].mean()
            data = data.set_index("NAME")
            
    
    return data
     


# In[32]:


base_model_regressor(X ,y, models, cv = 5)


# In[33]:


def hyperparameter_optimization(X , y , models, cv = 3):
    
    models_dict = {}
    data = pd.DataFrame()
    index = 0
    
    for name,model,params in models:
        
        cv_results = cross_validate(estimator=model,
                                 X = X,
                                 y = y,
                                 cv = cv,
                                 scoring = ["roc_auc","accuracy"])
        
        roc_auc = cv_results["test_roc_auc"].mean()
        accuracy = cv_results["test_accuracy"].mean()
        
        print("".center(50,"#"),end = "\n\n")
        print(f"For {name.upper()} Before CV :\n\nROC AUC : {roc_auc}\nAccuracy : {accuracy}",end = "\n\n")
        print("".center(50,"#"),end = "\n\n")
            
        best_grid = GridSearchCV(estimator=model,
                                 param_grid=params,
                                 cv=cv,
                                 scoring="roc_auc",
                                 n_jobs=-1,
                                 verbose=0).fit(X,y)
        
        final_model = model.set_params(**best_grid.best_params_)
        
        final_cv_results = cross_validate(estimator=final_model,
                                          X = X,
                                          y = y,
                                          cv = cv,
                                          scoring=["roc_auc","accuracy"])
        
        final_roc_auc = final_cv_results["test_roc_auc"].mean()
        final_accuracy = final_cv_results["test_accuracy"].mean()
        
        print(f"For {name.upper()} After CV : \n\nROC AUC : {final_roc_auc}\nAccuracy : {final_accuracy}",
              end = "\n\n")
        print(f"\n\nFor {name.upper()} Best Params : {best_grid.best_params_}",end = "\n\n")
        print("".center(50,"#"),end = "\n\n")
        
        models_dict[name] = final_model
        
    return models_dict
        


# In[34]:


from warnings import filterwarnings
filterwarnings("ignore")

model_dict = hyperparameter_optimization(X, y , models, cv = 10)


# In[35]:


for name,regressor,params in models:
    
    model_dict[name].fit(X,y)
    pd.to_pickle(model_dict[name], open("Final_"+name+"_Model.pkl","wb"))


# ## Final Model

# In[36]:


titanic_model = pickle.load(open("Final_LR_Model.pkl","rb"))


# In[37]:


titanic_model.get_params()


# In[38]:


def roc_auc_plot(model, X, y):
    
    y_score = model.predict_proba(X)[:,1] 
    
    fpr, tpr, thresholds = roc_curve(y,y_score)
    
    plt.figure(figsize = (7,5))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1] ,"--r")
    size = 15
    plt.xlabel(" False Positive Rate ", fontsize = size)
    plt.ylabel(" True Positive Rate ", fontsize = size)
    plt.title(" AUC (Area : %.3f)" % roc_auc_score(y, y_score) , fontsize = size)
    plt.show()
    
def confusion_matrix_heat(model, X ,y):
    
    y_pred = model.predict(X)
    data = confusion_matrix(y,y_pred)
    
    
    fig , axes = plt.subplots(figsize = (7,5))
 
    axes = sns.heatmap(data , annot = True , annot_kws= {"size" : 23} , fmt = ".3g", linewidths=3,
                      linecolor="white", cbar = False, cmap = "viridis")
    
    size = 15
    plt.xlabel(" Predict Label ", fontsize = size)
    plt.ylabel(" True Label ", fontsize = size)
    plt.title(f" For {type(model).__name__.upper()} Confusion Matrix ", fontsize = size)
    plt.show()
    
    


# In[39]:


confusion_matrix_heat(titanic_model, X, y)


# In[40]:


roc_auc_plot(titanic_model, X, y)


# In[ ]:




