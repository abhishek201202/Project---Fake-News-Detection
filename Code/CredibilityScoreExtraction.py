import pandas as pd

class CredibilityScoreExtraction:
    def __init__(self, df, textColumnName):
        print("Credibility Starts")
        self.m_new_df = pd.DataFrame()
        self.m_df = df.copy()
        self.score = {}
        self.m_textColumnName = textColumnName
        self.m_df[self.m_textColumnName] = [text.lower() for text in self.m_df[self.m_textColumnName]]
        self.CredibilityScore()
        self.MakeDataFrame()
        print("Credibility Done\n")
        
    def GetDataFrame(self):
        return self.m_new_df
    
    def CredibilityScore(self):
        unique = list(set(self.m_df[self.m_textColumnName]))
        label_dic = {
            "true" : 0,
            "mostly-true" : 1,
            "half-true" : 2,
            "barely-true" : 3,
            "false" : 4,
            "pants-fire" : 5
        }
        for id in unique:
            self.score[id] = label_dic.copy()
        
        for idx, val in self.m_df.iterrows():
            id = val[self.m_textColumnName]
            lb = val['label']
            self.score[id][lb] += 1
            
        for key in self.score:
            total = sum(self.score[key].values())
            self.score[key]['true'] /= total
            self.score[key]['mostly-true'] /= total
            self.score[key]['half-true'] /= total
            self.score[key]['barely-true'] /= total
            self.score[key]['false'] /= total
            self.score[key]['pants-fire'] /= total

    def MakeDataFrame(self):
        temp_df = []
        for idx, val in self.m_df.iterrows():
            temp_df.append(self.score[val[self.m_textColumnName]])
        self.m_new_df = pd.DataFrame.from_dict(temp_df)
