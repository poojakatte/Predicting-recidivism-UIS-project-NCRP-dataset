
# coding: utf-8

# # Predicting Recidivsm using Machine Learning

# Done By: Sahil Phule and Pooja Katte

# In[222]:

get_ipython().magic('matplotlib inline')
import mpld3
mpld3.enable_notebook()


# In[223]:

from pylab import rcParams
rcParams['figure.figsize'] = 12, 5


# In[9]:

df_pa = spark.read.option("delimiter","\t").option("header","true")     .csv("/project/data/ICPSR_36404/pris_admis/36404-0002-Data.tsv")


# In[10]:

df_tr = spark.read.option("delimiter","\t").option("header","true")     .csv("/project/data/ICPSR_36404/term_rec/36404-0001-Data.tsv")


# In[11]:

df_pr = spark.read.option("delimiter","\t").option("header","true")     .csv("/project/data/ICPSR_36404/pris_rel/36404-0003-Data.tsv")


# In[12]:

df_yep = spark.read.option("delimiter","\t").option("header","true")     .csv("/project/data/ICPSR_36404/year_end_pop/36404-0004-Data.tsv")


# In[13]:

df_tr.createOrReplaceTempView("term_rec")


# In[14]:

spark.sql("select * from term_rec limit 1")


# In[15]:

spark.sql("select count(*) from term_rec").show()


# # Counting values and plotting graph
# ## In term records

# ### For ADMTYPE

# In[16]:

spark.sql("select ADMTYPE, count(1)as num from term_rec group by ADMTYPE ").show()


# In[144]:

admtype = spark.sql("select ADMTYPE, count(1)as num from term_rec group by ADMTYPE ").toPandas()


# In[145]:

admtype


# In[149]:

admtype.plot(x=admtype.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Missing Values are: 295414

# ### For OFFGENERAL

# In[20]:

offgeneral = spark.sql("select OFFGENERAL, count(1)as num from term_rec group by OFFGENERAL ").toPandas()


# In[21]:

offgeneral


# In[22]:

offgeneral.plot(x=offgeneral.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Missing Value are 71108

# ### For EDUCATION

# In[23]:

education = spark.sql("select EDUCATION, count(1)as num from term_rec group by EDUCATION ").toPandas()


# In[24]:

education


# All the education values are missing

# ### For ADMITYR

# In[25]:

admityr = spark.sql("select ADMITYR, count(1)as num from term_rec group by ADMITYR order by ADMITYR ").toPandas()


# In[26]:

admityr


# In[27]:

admityr.columns[1]


# In[28]:

admityr.plot(x=admityr.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[29]:

admityr.loc[admityr['ADMITYR']=='9999']


# Missing Values are 496

# ### For RELEASEYR

# In[30]:

releaseyr = spark.sql("select RELEASEYR, count(1)as num from term_rec group by RELEASEYR order by RELEASEYR ").toPandas()


# In[31]:

releaseyr


# In[32]:

releaseyr.plot(x=releaseyr.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).figsize(10, 6).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:

releaseyr.loc[releaseyr['RELEASEYR']=='9999']


# ### For MAND_PRISREL_YEAR

# In[ ]:

mand_prisrel_year = releaseyr = spark.sql("select MAND_PRISREL_YEAR, count(1)as num from term_rec group by MAND_PRISREL_YEAR order by MAND_PRISREL_YEAR").toPandas()


# In[ ]:

mand_prisrel_year


# In[ ]:

mand_prisrel_year.plot(x=mand_prisrel_year.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[ ]:




# ### For PROJ_PRISREL_YEAR

# In[33]:

proj_prisrel_year = releaseyr = spark.sql("select PROJ_PRISREL_YEAR, count(1)as num from term_rec group by PROJ_PRISREL_YEAR order by PROJ_PRISREL_YEAR").toPandas()


# In[34]:

proj_prisrel_year


# In[35]:

proj_prisrel_year.plot(x=proj_prisrel_year.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# ### for PARELIG_YEAR

# In[36]:

parelig_year = spark.sql("select PARELIG_YEAR, count(1)as num from term_rec group by PARELIG_YEAR order by PARELIG_YEAR").toPandas()


# In[37]:

parelig_year


# In[38]:

parelig_year.plot(x=parelig_year.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# ### For SENTLGTH

# In[39]:

sentlgth = spark.sql("select SENTLGTH, count(1)as num from term_rec group by SENTLGTH order by SENTLGTH").toPandas()


# In[40]:

sentlgth


# In[41]:

sentlgth.plot(x=sentlgth.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Missing values are 20063 + 54469 = 74532

# ### For OFFDETAIL

# In[42]:

offdetail = spark.sql("select OFFDETAIL, count(1)as num from term_rec group by OFFDETAIL order by OFFDETAIL*1").toPandas()


# In[43]:

offdetail


# In[44]:

offdetail.plot(x=offdetail.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Missing  value are: 71108

# ### For RACE

# In[45]:

race = spark.sql("select RACE, count(1)as num from term_rec group by RACE order by RACE").toPandas()


# In[46]:

race


# In[47]:

race.plot(x=race.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Missing Vales are 1180982

# ### For AGEADMIT

# In[48]:

ageadmit = spark.sql("select AGEADMIT, count(1)as num from term_rec group by AGEADMIT order by AGEADMIT").toPandas()


# In[49]:

ageadmit


# In[50]:

ageadmit.plot(x=ageadmit.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# No Missing values

# ### For AGERELEASE

# In[51]:

agerelease = spark.sql("select AGERELEASE, count(1)as num from term_rec group by AGERELEASE order by AGERELEASE").toPandas()


# In[52]:

agerelease


# In[53]:

agerelease.plot(x=agerelease.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# No missing values apart from blank values

# ### For TIMESRVD

# In[54]:

timesrvd = spark.sql("select TIMESRVD, count(1)as num from term_rec group by TIMESRVD order by TIMESRVD").toPandas()


# In[55]:

timesrvd


# In[56]:

timesrvd.plot(x=timesrvd.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# Missing values are 1200886

# In[ ]:




# ### For RELTYPE

# In[57]:

reltype = spark.sql("select RELTYPE, count(1)as num from term_rec group by RELTYPE order by RELTYPE").toPandas()


# In[58]:

reltype


# In[60]:

reltype.plot(x=reltype.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# No missing Values

# ### For STATE

# In[61]:

state = spark.sql("select STATE, count(1)as num from term_rec group by STATE order by STATE*1").toPandas()


# In[62]:

state.plot(x=state.columns[0], kind='bar', color='Red', title="Number of Values", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# ______________________________________________________________________

# ### dropping columns MAND_PRISREL_YEAR, PROJ_PRISREL_YEAR,  PARELIG_YEAR,AGERELEASE as they contain a lot of missing values also dropping education as all values are missing

# In[63]:

spark.sql("select ABT_INMATE_ID,SEX,ADMTYPE,OFFGENERAL,ADMITYR,RELEASEYR,SENTLGTH,OFFDETAIL,RACE,AGEADMIT,TIMESRVD,RELTYPE,STATE from term_rec").createOrReplaceTempView("term_rec_clV1")


# ### removing missing data
# 

# In[64]:

spark.sql("select * from term_rec_clV1 where RELEASEYR != '9999' AND ADMTYPE != '9' AND OFFGENERAL != '9' AND ADMITYR != '9999' AND OFFDETAIL != '99' AND RACE != '9' AND AGEADMIT != '9' AND OFFDETAIL != '9' AND SENTLGTH != '9' AND RELTYPE != '9' AND TIMESRVD != '9' AND RELTYPE != ' ' AND SENTLGTH != ' '").createOrReplaceTempView("term_rec_clean")


# In[65]:

spark.sql("select count(*) from term_rec_clean").show()


# ### creating rep off and class

# In[66]:

spark.sql("select ABT_INMATE_ID ,count(*) as num from term_rec_clean group by ABT_INMATE_ID").createOrReplaceTempView("rep_off")


# In[67]:

spark.sql("select * from rep_off").show()


# In[68]:

spark.sql("select count (*) from rep_off").show()


# In[69]:

spark.sql("select *, case when num = '1' then 0 else 1 end as class from rep_off").createOrReplaceTempView("rep_off_class")


# In[70]:

spark.sql("select * from rep_off_class where ABT_INMATE_ID = 'A132015000000535415'").show()


# In[71]:

spark.sql("select * from term_rec_clean where ABT_INMATE_ID='A132015000000535415'").show()


# ### adding rep offence and class column

# In[72]:

spark.sql("select a.SEX,a.ADMTYPE,a.OFFGENERAL,a.ADMITYR,a.RELEASEYR,a.SENTLGTH,a.OFFDETAIL,a.RACE,a.AGEADMIT,a.TIMESRVD,a.RELTYPE,a.STATE,b.num AS repoff, b.class from term_rec_clean a left join rep_off_class b on (a.ABT_INMATE_ID = b.ABT_INMATE_ID)").createOrReplaceTempView("pris_data")


# In[129]:

spark.sql("select * from pris_data ").show()


# In[73]:

spark.sql("select count(*) from pris_data ").show()


# In[74]:

pris_data_converted = spark.sql("select cast(SEX as int),cast(ADMTYPE as int),cast(OFFGENERAL as int),cast(ADMITYR as int),cast(RELEASEYR as int),cast(SENTLGTH as int),cast(OFFDETAIL as int),cast(RACE as int),cast(AGEADMIT as int),cast(TIMESRVD as int),cast(RELTYPE as int),cast(STATE as int),cast(class as double) as label from pris_data ")


# In[75]:

pris_data_converted.printSchema()


# ## Transforming Data

# In[76]:

#from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

#label_indexer = StringIndexer(inputCol = 'class', outputCol = 'label')
#plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')

numeric_cols = ["SEX","ADMTYPE","OFFGENERAL","ADMITYR","RELEASEYR","SENTLGTH",
                "OFFDETAIL","RACE","AGEADMIT","TIMESRVD","RELTYPE","STATE"]

assembler = VectorAssembler( inputCols = numeric_cols,outputCol = 'features')


# ## Splitting into train and test data 

# In[77]:

train, test = pris_data_converted.randomSplit([0.7, 0.3], seed=42)


# ## Predictions using logistic regression

# In[78]:

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,featuresCol = 'features')
pipeline_lr = Pipeline(stages=[assembler, lr])
model_lr=pipeline_lr.fit(train)


# In[79]:

predictions_lr=model_lr.transform(test)


# In[82]:

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
auroc_lr = evaluator.evaluate(predictions_lr, {evaluator.metricName: "areaUnderROC"})


# In[83]:

auroc_lr


# ## Predictions using random forest

# In[84]:


from pyspark.ml.classification import RandomForestClassifier

#classifier = RandomForestClassifier(labelCol = 'class', featuresCol = 'features')
classifier = RandomForestClassifier(featuresCol = 'features')
pipeline = Pipeline(stages=[assembler, classifier])
model = pipeline.fit(train)


# In[85]:

from pyspark.ml.evaluation import BinaryClassificationEvaluator
 
predictions = model.transform(test)


# In[86]:

#evaluator = BinaryClassificationEvaluator(labelCol='class')
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})


# In[87]:

auroc


# In[ ]:




# ## second method

# In[88]:

modelv2 = classifier.fit(assembler.transform(train))


# In[89]:

predictionsv2 = modelv2.transform(assembler.transform(test))


# In[90]:

auroc = evaluator.evaluate(predictionsv2, {evaluator.metricName: "areaUnderROC"})


# In[91]:

auroc


# In[92]:

modelv2.featureImportances


# ### Plotting graph of  features importance

# In[174]:




# In[196]:

import pandas as pd
import numpy as np
cols = np.array(['SEX',
'ADMTYPE',
'OFFGENERAL',
'ADMITYR',
'RELEASEYR',
'SENTLGTH',
'OFFDETAIL',
'RACE',
'AGEADMIT',
'TIMESRVD',
'RELTYPE',
'STATE'])

tempdf = pd.DataFrame(modelv2.featureImportances.toArray().tolist())


# In[200]:

tempdf['cols'] = cols


# In[201]:

tempdf


# In[225]:

tempdf.plot(x= tempdf['cols'], kind='bar', color='Red', title="Feature Importance", grid=True).legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[538]:

predictions


# In[133]:

predictions.select("rawPrediction","probability","prediction").show()


# In[ ]:




# In[ ]:




# In[ ]:




# ## Re creating the class variable 

# In[93]:

spark.sql("select ABT_INMATE_ID,ADMITYR, count(*) as num from term_rec_clean group by ABT_INMATE_ID,ADMITYR").createOrReplaceTempView("rep_off_v2")


# In[94]:

spark.sql("select * from rep_off_v2").show()


# In[95]:

spark.sql("select * from rep_off_v2 where ABT_INMATE_ID = 'A132015000000535415'").show()


# In[96]:

spark.sql("select * from rep_off_v2 where ABT_INMATE_ID = 'A272015000000081052'").show()


# In[97]:

spark.sql("select ABT_INMATE_ID,count (*) as num from rep_off_v2 group by ABT_INMATE_ID ").createOrReplaceTempView("rep_off_v2_actual")


# In[ ]:




# In[98]:

spark.sql("select * from rep_off_v2_actual where ABT_INMATE_ID = 'A132015000000535415'").show()


# #### assignining class column to the corrected inmate column

# In[99]:

spark.sql("select *, case when num = '1' then 0 else 1 end as class from rep_off_v2_actual").createOrReplaceTempView("rep_off_v2_actual_class")


# In[100]:

spark.sql("select * from rep_off_v2_actual_class where ABT_INMATE_ID = 'A132015000000535415'").show()


# In[101]:

spark.sql("select * from term_rec_clean where ABT_INMATE_ID='A132015000000535415'").show()


# In[ ]:




# In[ ]:




# In[ ]:




# ### adding corrected rep offence and class column

# In[103]:

spark.sql("select a.SEX,a.ADMTYPE,a.OFFGENERAL,a.ADMITYR,a.RELEASEYR,a.SENTLGTH,a.OFFDETAIL,a.RACE,a.AGEADMIT,a.TIMESRVD,a.RELTYPE,a.STATE,b.num AS repoff, b.class from term_rec_clean a left join rep_off_v2_actual_class b on (a.ABT_INMATE_ID = b.ABT_INMATE_ID)").createOrReplaceTempView("pris_data_v2")


# In[104]:

spark.sql("select count(*) from pris_data_v2 ").show()


# In[105]:

pris_data_v2_converted = spark.sql("select cast(SEX as int),cast(ADMTYPE as int),cast(OFFGENERAL as int),cast(ADMITYR as int),cast(RELEASEYR as int),cast(SENTLGTH as int),cast(OFFDETAIL as int),cast(RACE as int),cast(AGEADMIT as int),cast(TIMESRVD as int),cast(RELTYPE as int),cast(STATE as int),cast(class as double) as label from pris_data_v2 ")


# In[106]:

pris_data_v2_converted.printSchema()


# ## Transforming Data

# In[107]:

#from pyspark.ml.feature import StringIndexer
#from pyspark.ml.feature import VectorAssembler

#label_indexer = StringIndexer(inputCol = 'class', outputCol = 'label')
#plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')

numeric_cols = ["SEX","ADMTYPE","OFFGENERAL","ADMITYR","RELEASEYR","SENTLGTH",
                "OFFDETAIL","RACE","AGEADMIT","TIMESRVD","RELTYPE","STATE"]

assembler = VectorAssembler( inputCols = numeric_cols,outputCol = 'features')


# ## Splitting new data into train and test data 

# In[108]:

train_v2, test_v2 = pris_data_v2_converted.randomSplit([0.7, 0.3], seed=42)


# ## Predictions using logistic regression for new data

# In[586]:

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,featuresCol = 'features')
pipeline_lr_v2 = Pipeline(stages=[assembler, lr])
model_lr_v2=pipeline_lr_v2.fit(train_v2)


# In[587]:

predictions_lr_v2=model_lr_v2.transform(test_v2)


# In[588]:

auroc_lr_v2 = evaluator.evaluate(predictions_lr_v2, {evaluator.metricName: "areaUnderROC"})


# In[589]:

auroc_lr_v2


# ## Predictions using random forest for new data

# In[109]:




#classifier = RandomForestClassifier(labelCol = 'class', featuresCol = 'features')
classifier = RandomForestClassifier(featuresCol = 'features')
pipeline = Pipeline(stages=[assembler, classifier])
model_v2 = pipeline.fit(train_v2)


# In[110]:

from pyspark.ml.evaluation import BinaryClassificationEvaluator
 
predictions_v2 = model_v2.transform(test_v2)


# In[111]:

#evaluator = BinaryClassificationEvaluator(labelCol='class')
evaluator = BinaryClassificationEvaluator()
auroc_v2 = evaluator.evaluate(predictions_v2, {evaluator.metricName: "areaUnderROC"})


# In[112]:

auroc_v2


# ## Calculating average average time that it took inmates to return to prison after release

# Here only 3 colums are required: iname id, admit year, release year

# In[113]:

spark.sql("select ABT_INMATE_ID,ADMITYR,RELEASEYR from term_rec_clean group by ABT_INMATE_ID,ADMITYR,RELEASEYR order by ADMITYR,RELEASEYR ").createOrReplaceTempView("avg_time")


# In[114]:

spark.sql("select *, row_number() over (partition by ABT_INMATE_ID order by ADMITYR,RELEASEYR ) as idx from avg_time ").createOrReplaceTempView("avg_time_idx")


# In[115]:

pris_adm_rel = spark.sql("select * from avg_time_idx where ABT_INMATE_ID = 'A272015000000081052' OR ABT_INMATE_ID = 'A132015000000535415' order by ABT_INMATE_ID,idx")


# In[116]:

pris_adm_rel.show()


# In[117]:

pris_adm_rel.show()


# Now that we have created the ids, we will select all the elements

# In[118]:

pris_adm_rel = spark.sql("select * from avg_time_idx ")


# In[128]:

spark.sql("select * from avg_time_idx where ABt_INMATE_ID = 'A492015000000017034'").show()


# In[119]:

pris_adm_rel.show()


# In[121]:

pris_admrel_rdd = pris_adm_rel.rdd


# In[122]:

pris_admrel_rdd.map(tuple).groupBy(lambda a: a[0]).mapValues(lambda xs: [(x) for x in xs]).take(5)


# In[123]:

grp_id = pris_admrel_rdd.map(tuple).groupBy(lambda a: a[0]).mapValues(lambda xs:[(x[3],int(x[1]),int(x[2]),x[3]+1) for x in xs if len(xs)>1]).filter(lambda v: len(v[1])>1)


# In[162]:

grp_id.take(5)


# In[124]:

grp_id2= grp_id.mapValues(lambda xs:[(y[1]-x[2]) for x in xs for y in xs  if x[3]==y[0] ])


# In[125]:

grp_id2.take(5)


# ## Average

# In[127]:

grp_id2.mapValues(lambda av:sum(av)/len(av) ).map(lambda std:std[1] ).mean()


# ## Standard deviation

# In[126]:

grp_id2.mapValues(lambda av:sum(av)/len(av) ).map(lambda std:std[1] ).stdev()


# In[ ]:



