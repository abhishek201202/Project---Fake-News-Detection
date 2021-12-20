import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te

class SentimentFeatureExtraction:
    def __init__(self, uncleaned, textColumnName):
        print("Sentiment Feature Extraction Starts")
        self.m_analyzer = SentimentIntensityAnalyzer()
        self.m_new_df = pd.DataFrame()
        self.m_uncleaned = uncleaned
        self.m_textColumnName = textColumnName
        self.Sentiment()
        print("Sentiment Feature Extraction Done\n")
        
    def GetDataFrame(self):
        return self.m_new_df
    
    def Sentiment(self):
        sentiment = [self.m_analyzer.polarity_scores(text[self.m_textColumnName]) for idx,text in self.m_uncleaned.iterrows()]
        self.m_new_df = pd.DataFrame.from_dict(sentiment)
        self.m_new_df.drop(['compound'], axis='columns',inplace=True)

    def TextEmotion(self):
        emotion = [te.get_emotion(text[self.m_textColumnName]) for idx,text in self.m_uncleaned.iterrows()]
        emotion = pd.DataFrame.from_dict(emotion)
        emotion.reset_index(drop=True, inplace=True)
        self.m_new_df.reset_index(drop=True, inplace=True)
        self.m_new_df = pd.concat([self.m_new_df, emotion], axis=1)