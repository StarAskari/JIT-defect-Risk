import re
import string
import pandas as pd

class CommitMessageCleaner:
    """
    Clean and normalize commit messages for NLP processing
    """
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.issue_pattern = re.compile(r'#\d+')

    def remove_urls(self, text):
        """ remove urls from text """ 
        return self.url_pattern.sub('', text)
    
    def remove_emails(self, text):
        """ remove emails from text """
        return self.email_pattern.sub('', text)
    
    def remove_control_chars(self, text):
        """ remove control characters """
        return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    def remove_code_blocks(self , text):
        """Remove code snippets in backticks"""
        # Remove code blocks with triple backticks
        text = re.sub(r'```[\s\S]*?```', '', text)
        # Remove inline code with single backticks
        text = re.sub(r'`[^`]*`', '', text)
        return text
    
    def normalize_whitespace(self, text):
        """Normalize whitespace - replace multiple spaces/newlines with single space"""
        # Replace newlines and tabs with spaces
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        #replace multiple space with one space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_special_tokens(self, text):
        #remove git specific tokens and patterns
        patterns = [
            r'Signed-off-by:.*',
            r'Co-authored-by:.*',
            r'Reviewed-by:.*',
            r'Tested-by:.*',
            r'Acked-by:.*',
            r'Fixes:.*',
            r'Closes:.*',
            r'Resolves:.*'
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    def preserve_issue_references(self, text):
        """
        Convert issue references (#123) to a token
        This helps the model learn that issues are important
        """
        return self.issue_pattern.sub('ISSUE_REF', text)
    
    def clean_commit_message(self, message, preserve_issues = True):
        if not isinstance(message, str):
            return ""
        message = self.remove_urls(message)
        message = self.remove_emails(message)
        message = self.remove_code_blocks(message)
        message = self.remove_control_chars(message)
        message = self.remove_special_tokens(message)
        
        if preserve_issues:
            message = self.preserve_issue_references(message)
        else:
            message = self.issue_pattern.sub('', message)
        
        message = self.normalize_whitespace(message)
        message = message.lower()
        return message
    
    def clean_dataset(self, df, message_column='message'):
        print(f"Cleaning {len(df)} commit messages...")
        df['clean_message'] = df[message_column].apply(self.clean_commit_message)

        original_len = len(df)
        df = df[df['clean_message'].str.len()>0]
        removed = original_len - len(df)

        if removed > 0:
            print(f"Removed {removed} commits with empty messages after cleaning")

        avg_length = df['clean_message'].str.len().mean()
        avg_words =  df['clean_message'].str.split().str.len().mean()

        print(f"Average message length: {avg_length: .1f} characteres")
        print(f"Average word count: {avg_words:.1f} words")
        return df

if __name__ == "__main__":
    
    
        # Test with sample messages
        test_messages = [
            "Fix bug in authentication module (#1234)\n\nSigned-off-by: John Doe <john@example.com>",
            "Add new feature for user management\n\nThis commit adds:\n```python\ndef foo():\n    pass\n```\n\nFixes #456",
            "Update documentation\n\nCo-authored-by: Jane Smith <jane@example.com>",
        ]
    
        cleaner = CommitMessageCleaner()
    
        print("=== Testing Text Cleaner ===\n")
        for i, msg in enumerate(test_messages, 1):
            print(f"Original {i}:")
            print(msg)
            print(f"\nCleaned {i}:")
            print(cleaner.clean_commit_message(msg))
            print("-" * 80)   
