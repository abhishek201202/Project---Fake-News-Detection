from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import string, re

class TextCleaning:
    def __init__(self, df, textColumnName):
        self.m_df = df.copy()
        self.m_textColumnName = textColumnName
        self.m_stopWords = list(stopwords.words('english'))
        self.m_lemmatizer = WordNetLemmatizer()
        self.m_stemmer = PorterStemmer()
        print("Text Cleaning Starts")
        self.LowerText()
        print("\t\tText Lowered")
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanPunctuation)
        print("\t\tPunctuation Removed")
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanStopWords)
        print("\t\tStopwords Removed")
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanUrls)
        print("\t\tUrls Removed")
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanHashTags)
        print("\t\tHashTags Removed")
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanNumbers)
        print("\t\tNumbers Removed")
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.LemmatizeText)
        print("\t\tLemmatization Done")
        print("Text Cleaning Done\n")
    
    def GetDataFrame(self):
        return self.m_df
    
    def LowerText(self):
        self.m_df[self.m_textColumnName] = [text.lower() for text in self.m_df[self.m_textColumnName]]

    def CleanPunctuation(self, text):
        translationTable = text.maketrans('', '', string.punctuation)
        return text.translate(translationTable)

    def CleanStopWords(self, text):
        wordsInText = text.split()
        wordsInText = [w for w in wordsInText if not w in self.m_stopWords]
        return ' '.join(wordsInText)

    def CleanUrls(self, text):
        return re.sub("http\S+", "", text)
    
    def CleanHashTags(self, text):
        return re.sub("#\S+", "", text)
    
    def CleanNumbers(self, text):
        return re.sub("\d+", "", text)

    def LemmatizeText(self, text):
        return ' '.join([self.m_lemmatizer.lemmatize(word) for word in text.split()])
    
    def StemmingText(self, text):
        return ' '.join([self.m_stemmer.stem(word) for word in text.split()])