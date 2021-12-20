import seaborn as sns

class DataVisualization:
    def __init__(self, df, textColumnName):
        self.m_df = df
        self.m_textColumnName = textColumnName
        
    ## function to check distribution of labels
    def check_distribution(self):
        sns.countplot(self.m_textColumnName, data=self.m_df, palette='hls')