import pandas as pd
import spacy
from collections import Counter

class DependencyFeatureExtraction:
    def __init__(self, df, textColumnName):
        print("Dependency Feature Extraction Starts")
        self.m_new_df = pd.DataFrame()
        self.m_df = df
        self.m_textColumnName = textColumnName
        self.m_Dep = spacy.load("en_core_web_sm")
        self.m_Dep_features = self.m_Dep.pipe_labels['parser']
        self.Dependency()
        print("Dependency Feature Extraction Done\n")
        
    def GetDataFrame(self):
        return self.m_new_df
        
    def Dependency(self):
        dependencies = []
        for idx, row in self.m_df.iterrows():
            sentence = self.m_Dep(row[self.m_textColumnName])
            dic = dict.fromkeys(self.m_Dep_features,0)
            labels = [x.dep_ for x in sentence]
            labels = Counter(labels)
            for key in labels.keys():
                if key in dic:
                    dic[key] += labels[key]
            dependencies.append(dic)
        dependencies_df = pd.DataFrame.from_dict(dependencies)
        self.m_new_df = dependencies_df
