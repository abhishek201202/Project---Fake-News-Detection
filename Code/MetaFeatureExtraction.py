import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize

class MetaFeatureExtraction:
    def __init__(self, df, textColumnName):
        print("Meta Feature Extraction Starts")
        self.m_new_df = pd.DataFrame()
        self.m_df = df
        self.m_textColumnName = textColumnName
        self.MetaFeature()
        print("Meta Feature Extraction Done\n")
        
    def GetDataFrame(self):
        return self.m_new_df

    def MetaFeature(self):
        self.m_new_df['total_sentences'] = self.m_df[self.m_textColumnName].apply( lambda text: len(sent_tokenize(text)))
        self.m_new_df['number_of_words'] = self.m_df[self.m_textColumnName].apply( lambda text: len(word_tokenize(text)))
        self.m_new_df['number_of_unique_words'] = self.m_df[self.m_textColumnName].apply( lambda text: len(set(word_tokenize(text))))
        self.m_new_df['number_of_characters'] = self.m_df[self.m_textColumnName].apply(lambda text: len(text))
        self.m_new_df['number of characters per words'] = self.m_new_df['number_of_characters'] / self.m_new_df['number_of_words']
        self.m_new_df['avg_sentence_len'] = self.m_new_df['number_of_characters'] / self.m_new_df['total_sentences']
        self.m_new_df['avg_word_len'] = self.m_new_df['number_of_characters'] / self.m_new_df['number_of_words']
