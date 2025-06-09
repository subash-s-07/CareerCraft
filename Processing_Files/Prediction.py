import pandas as pd
import sys
import asyncio
sys.path.append(r"C:\Users\ADMIN\OneDrive\Desktop\MCP\NLP-Project\Web Scrapping")
import xgboost as xgb
model = xgb.XGBRegressor()
from Experience_Match import get_exp_score
from Job_Match import get_role_match_score
from Match_Skills import get_skill_score
from Sbert import similarity_score_process
from Job_Detail_WebScrape import main
def return_jobs(job_role):
    df=asyncio.run(main(job_role))
    data2=get_exp_score(df).drop(['Job_Description'],axis=1)
    data3=get_role_match_score(df).drop(['Job_Description'],axis=1)
    data4=get_skill_score(df).drop(['Job_Description'],axis=1)
    data5=similarity_score_process(df).drop(['Job_Description'],axis=1)
    data=pd.merge(df,data2,on='Job_ID')
    data=pd.merge(data,data3,on='Job_ID')
    data=pd.merge(data,data4,on='Job_ID')
    data=pd.merge(data,data5,on='Job_ID')
    model.load_model(r"C:\Users\ADMIN\OneDrive\Desktop\MCP\NLP-Project\xgb_model.json")
    X=data[['exp_score','role_match_score','skill_match_score','similarity_score']]
    data['Score']=model.predict(X)
    top_10 = data.sort_values(by='Score', ascending=False).head(10)
    return top_10

