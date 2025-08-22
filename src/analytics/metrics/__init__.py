"""Basic metrics implementation for analytics."""
from .semantic_difference import SemanticDifferenceMetric

class BasicMetrics:
    """
    Basic metrics for analytics.
    Contains factory methods that return metric functions.
    """
    
    @staticmethod
    def mean(column):
        """
        Return a function that calculates the mean of a column.
        
        Args:
            column: Column name to calculate mean for
            
        Returns:
            Function that takes a DataFrame and returns the mean value
        """
        def mean_func(df):
            if column not in df.columns:
                return None
            return df[column].mean()
        return mean_func
    
    @staticmethod
    def count():
        """
        Return a function that counts rows in a DataFrame.
        
        Returns:
            Function that takes a DataFrame and returns the row count
        """
        def count_func(df):
            return len(df)
        return count_func
    
    @staticmethod
    def sum(column):
        """
        Return a function that calculates the sum of a column.
        
        Args:
            column: Column name to calculate sum for
            
        Returns:
            Function that takes a DataFrame and returns the sum
        """
        def sum_func(df):
            if column not in df.columns:
                return None
            return df[column].sum()
        return sum_func
    
    @staticmethod
    def min(column):
        """
        Return a function that calculates the minimum of a column.
        
        Args:
            column: Column name to calculate minimum for
            
        Returns:
            Function that takes a DataFrame and returns the minimum value
        """
        def min_func(df):
            if column not in df.columns:
                return None
            return df[column].min()
        return min_func
    
    @staticmethod
    def max(column):
        """
        Return a function that calculates the maximum of a column.
        
        Args:
            column: Column name to calculate maximum for
            
        Returns:
            Function that takes a DataFrame and returns the maximum value
        """
        def max_func(df):
            if column not in df.columns:
                return None
            return df[column].max()
        return max_func