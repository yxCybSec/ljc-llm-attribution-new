import pandas as pd
import email
import py3langid
from typing import Tuple, List


class DataLoader:
    @staticmethod
    def load_blog_data(filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        df.drop(['gender', 'age', 'topic', 'sign', 'date'], axis=1, inplace=True)
        df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)
        df['lang'] = df['text'].apply(lambda x: py3langid.classify(x)[0])
        df = df[df.lang == 'en']
        df.drop('lang', axis=1, inplace=True)

        v = df.id.value_counts()
        df = df[df.id.isin(v[v >= 2].index)]

        return df

    @staticmethod
    def load_email_data(filepath: str) -> pd.DataFrame:
        emails_df = pd.read_csv(filepath)

        def get_text_from_email(msg):
            parts = []
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    parts.append(part.get_payload())
            return ''.join(parts)

        def split_email_addresses(line):
            if line:
                addrs = line.split(',')
                addrs = frozenset(map(lambda x: x.strip(), addrs))
                return addrs
            return None

        messages = list(map(email.message_from_string, emails_df['message']))

        for key in messages[0].keys():
            emails_df[key] = [doc[key] for doc in messages]

        emails_df['Text'] = list(map(get_text_from_email, messages))
        emails_df['From'] = emails_df['From'].map(split_email_addresses)
        emails_df['To'] = emails_df['To'].map(split_email_addresses)

        del messages

        emails_df = emails_df[['From', 'To', 'Text', 'Date', 'message']]
        emails_df = emails_df.drop_duplicates(subset=['Text'], keep='first').reset_index(drop=True)

        mail_corpus = emails_df.copy()
        mail_corpus.columns = ['user', 'receiver', 'text', 'date', 'message_old']

        unique_author = mail_corpus['user'].unique()
        email_mapping = {k: v for k, v in zip(unique_author, range(len(unique_author)))}
        mail_corpus['id'] = mail_corpus['user'].apply(lambda x: 'mail_' + str(email_mapping[x]))

        mail_corpus.text = mail_corpus.text.apply(lambda x: x.strip())

        return mail_corpus[['text', 'id']]