import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
from textblob import TextBlob
import emoji
from collections import Counter
import nltk
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def analyze_conversations(conversation_dir, contact_name):
    # List all CSV files in the specified directory that contain the contact name in the file name
    csv_files = [f for f in os.listdir(conversation_dir) if f.endswith('.csv') and contact_name in f]
    
    # Check if there are any CSV files to process
    if not csv_files:
        print(f"No CSV files found with the contact name '{contact_name}' in the file name.")
        return
    
    # Initialize an empty list to hold DataFrames
    data_frames = []
    
    # Loop through each CSV file and append to the list
    for csv_file in csv_files:
        file_path = os.path.join(conversation_dir, csv_file)
        df = pd.read_csv(file_path)
        data_frames.append(df)
    
    # Concatenate all DataFrames in the list
    all_conversations = pd.concat(data_frames, ignore_index=True)
    
    # Extract the full contact name from the 'Chat Session' column
    full_contact_name = all_conversations['Chat Session'].iloc[0]
    
    # Add a column to specify 'Me' or 'Them' based on the 'Type' column
    all_conversations['Sender'] = all_conversations['Type'].apply(lambda x: 'Me' if x == 'Outgoing' else full_contact_name)
    
    # Filter conversations by the given contact name
    df_filtered = all_conversations[['Sender', 'Text', 'Message Date']].copy()
    
    # Remove rows where 'Text' is NaN
    df_filtered = df_filtered.dropna(subset=['Text'])
    
    # Calculate the length of each message
    df_filtered['Text Length'] = df_filtered['Text'].apply(len)
    
    # Perform sentiment analysis using TextBlob
    df_filtered['Sentiment'] = df_filtered['Text'].apply(lambda text: TextBlob(text).sentiment.polarity)
    
    # Function to count emojis in a text
    def count_emojis(text):
        return sum(1 for char in text if char in emoji.EMOJI_DATA)
    
    # Count emojis in each message
    df_filtered['Emoji Count'] = df_filtered['Text'].apply(count_emojis)
    
    # Separate texts for word clouds
    texts_me = ' '.join(df_filtered[df_filtered['Sender'] == 'Me']['Text'])
    texts_them = ' '.join(df_filtered[df_filtered['Sender'] == full_contact_name]['Text'])
    texts_both = ' '.join(df_filtered['Text'])
    
    # Word frequency analysis
    word_freq_me = Counter(nltk.word_tokenize(texts_me.lower()))
    word_freq_them = Counter(nltk.word_tokenize(texts_them.lower()))
    word_freq_both = Counter(nltk.word_tokenize(texts_both.lower()))
    
    # Part-of-speech tagging
    df_filtered['POS'] = df_filtered['Text'].apply(lambda text: nltk.pos_tag(nltk.word_tokenize(text)))
    
    # Group by 'Sender' and sum the text lengths
    character_counts = df_filtered.groupby('Sender')['Text Length'].sum().reset_index()
    
    # Group by 'Sender' to count the number of messages
    message_counts = df_filtered['Sender'].value_counts().reset_index()
    message_counts.columns = ['Sender', 'Message Count']
    
    # Ensure both 'Me' and full_contact_name are in the DataFrame for character counts
    if 'Me' not in character_counts['Sender'].values:
        character_counts = pd.concat([character_counts, pd.DataFrame({'Sender': ['Me'], 'Text Length': [0]})], ignore_index=True)
    if full_contact_name not in character_counts['Sender'].values:
        character_counts = pd.concat([character_counts, pd.DataFrame({'Sender': [full_contact_name], 'Text Length': [0]})], ignore_index=True)
    
    # Ensure both 'Me' and full_contact_name are in the DataFrame for message counts
    if 'Me' not in message_counts['Sender'].values:
        message_counts = pd.concat([message_counts, pd.DataFrame({'Sender': ['Me'], 'Message Count': [0]})], ignore_index=True)
    if full_contact_name not in message_counts['Sender'].values:
        message_counts = pd.concat([message_counts, pd.DataFrame({'Sender': [full_contact_name], 'Message Count': [0]})], ignore_index=True)
    
    # Calculate average sentiment per sender
    average_sentiment = df_filtered.groupby('Sender')['Sentiment'].mean().reset_index()
    average_sentiment.columns = ['Sender', 'Average Sentiment']
    
    # Sum of emojis per sender
    emoji_counts = df_filtered.groupby('Sender')['Emoji Count'].sum().reset_index()
    emoji_counts.columns = ['Sender', 'Total Emojis']
    
    # Create a directory for the contact name
    output_dir = os.path.join(conversation_dir, contact_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Display the results
    print("Character Counts:\n", character_counts)
    print("Message Counts:\n", message_counts)
    print("Average Sentiment:\n", average_sentiment)
    print("Emoji Counts:\n", emoji_counts)
    print("Word Frequency Me:\n", word_freq_me.most_common(20))
    print("Word Frequency Them:\n", word_freq_them.most_common(20))
    print("Word Frequency Both:\n", word_freq_both.most_common(20))
    
    # Define consistent colors
    colors = {'Me': 'blue', full_contact_name: 'green'}
    
    # Plot the character counts
    plt.figure(figsize=(10, 6))
    plt.bar(character_counts['Sender'], character_counts['Text Length'], color=[colors[sender] for sender in character_counts['Sender']])
    plt.xlabel('Sender')
    plt.ylabel('Total Characters')
    plt.title(f'Character Count by {full_contact_name} and Me')
    plt.savefig(os.path.join(output_dir, 'character_counts.png'))
    plt.show()

    # Plot the message counts
    plt.figure(figsize=(10, 6))
    plt.bar(message_counts['Sender'], message_counts['Message Count'], color=[colors[sender] for sender in message_counts['Sender']])
    plt.xlabel('Sender')
    plt.ylabel('Number of Messages')
    plt.title(f'Message Count by {full_contact_name} and Me')
    plt.savefig(os.path.join(output_dir, 'message_counts.png'))
    plt.show()
    
    # Plot the average sentiment
    plt.figure(figsize=(10, 6))
    plt.bar(average_sentiment['Sender'], average_sentiment['Average Sentiment'], color=[colors[sender] for sender in average_sentiment['Sender']])
    plt.xlabel('Sender')
    plt.ylabel('Average Sentiment')
    plt.title(f'Average Sentiment by {full_contact_name} and Me')
    plt.savefig(os.path.join(output_dir, 'average_sentiment.png'))
    plt.show()

    # Plot the emoji counts
    plt.figure(figsize=(10, 6))
    plt.bar(emoji_counts['Sender'], emoji_counts['Total Emojis'], color=[colors[sender] for sender in emoji_counts['Sender']])
    plt.xlabel('Sender')
    plt.ylabel('Total Emojis')
    plt.title(f'Emoji Count by {full_contact_name} and Me')
    plt.savefig(os.path.join(output_dir, 'emoji_counts.png'))
    plt.show()
    
    # Generate word cloud for 'Me'
    wordcloud_me = WordCloud(width=800, height=400, background_color='white').generate(texts_me)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_me, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Me')
    plt.savefig(os.path.join(output_dir, 'wordcloud_me.png'))
    plt.show()
    
    # Generate word cloud for 'Them'
    wordcloud_them = WordCloud(width=800, height=400, background_color='white').generate(texts_them)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_them, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {full_contact_name}')
    plt.savefig(os.path.join(output_dir, 'wordcloud_them.png'))
    plt.show()
    
    # Generate word cloud for both
    wordcloud_both = WordCloud(width=800, height=400, background_color='white').generate(texts_both)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud_both, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Both {full_contact_name} and Me')
    plt.savefig(os.path.join(output_dir, 'wordcloud_both.png'))
    plt.show()
    
    # Sentiment over time
    df_filtered['Message Date'] = pd.to_datetime(df_filtered['Message Date'])
    df_filtered.set_index('Message Date', inplace=True)
    sentiment_over_time = df_filtered.resample('D')['Sentiment'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_over_time['Message Date'], sentiment_over_time['Sentiment'], marker='o', linestyle='-', color='purple')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment')
    plt.title(f'Sentiment Over Time for {full_contact_name} and Me')
    plt.savefig(os.path.join(output_dir, 'sentiment_over_time.png'))
    plt.show()
    

  
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python analyze_conversations.py <conversation_dir> <contact_name>")
        sys.exit(1)
    
    conversation_dir = sys.argv[1]
    contact_name = sys.argv[2]
    
    analyze_conversations(conversation_dir, contact_name)
