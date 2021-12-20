import pandas as pd
import spacy
from collections import Counter

class NERFeatureExtraction:
    def __init__(self, df, textColumnName):
        print("NER Feature Extraction Starts")
        self.m_new_df = pd.DataFrame()
        self.m_df = df
        self.m_textColumnName = textColumnName
        self.m_NER = spacy.load("en_core_web_sm")
        self.m_NER_features = ["PERSON","ORG","FAC","GPE","NORP","LOC","PRODUCT","EVENT","WORK_OF_ART","LAW","LANGUAGE",
                            "DATE","TIME","PERCENT","MONEY","CARDINAL","QUANTITY","ORDINAL"]
        self.NER()
        print("NER Feature Extraction Done\n")
        
    def GetDataFrame(self):
        return self.m_new_df
        
    def NER(self):
        ner = []
        for idx, row in self.m_df.iterrows():
            sentence = self.m_NER(row[self.m_textColumnName])
            dic = dict.fromkeys(self.m_NER_features,0)
            labels = [x.label_ for x in sentence.ents]
            dic.update(Counter(labels))
            ner.append(dic)
        ner_df = pd.DataFrame.from_dict(ner)
        self.m_new_df = ner_df
