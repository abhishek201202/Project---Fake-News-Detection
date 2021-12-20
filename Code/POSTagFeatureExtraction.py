import pandas as pd
import spacy
from collections import Counter

class POSTagFeatureExtraction:
    def __init__(self, df, textColumnName):
        print("POS Tag Feature Extraction Starts")
        self.m_new_df = pd.DataFrame()
        self.m_df = df
        self.m_textColumnName = textColumnName
        self.m_POS = spacy.load("en_core_web_sm")
        self.m_POS_features = [ "ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART",
                            "PRON","X","PROPN","PUNCT","SCONJ","SYM","VERB","SPACE","CONJ"]
        self.POS()
        print("POS Tag Feature Extraction Done\n")
        
    def GetDataFrame(self):
        return self.m_new_df
        
    def POS(self):
        pos_tag = []
        for idx, row in self.m_df.iterrows():
            sentence = self.m_POS(row[self.m_textColumnName])
            dic = dict.fromkeys(self.m_POS_features,0)
            labels = [x.pos_ for x in sentence]
            dic.update(Counter(labels))
            pos_tag.append(dic)
        pos_df = pd.DataFrame.from_dict(pos_tag)
        self.m_new_df = pos_df
