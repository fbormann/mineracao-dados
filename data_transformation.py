from sense2vec import Sense2Vec, Sense2VecComponent
import spacy, pandas

nlp = spacy.load("en_core_web_sm")
s2v = Sense2Vec().from_disk("./models/s2v_reddit_2015_md/s2v_old/")

df = pandas.read_csv("./twitter_data/exploration_dataset.csv")

vectors_df = pandas.DataFrame(columns=['id','vectors','label'])

for idx, row in df.iterrows():
    print("Parsing sentences")
    try:
        doc = nlp(row['text'])
        vectors = []
        for token in doc:
            key = "{0}|{1}".format(token.lemma_, token.pos_)
            if key in s2v:
                vector = s2v[key]
                vectors.append(vector)
        vectors_df.append({
            "id":idx,
            "vectors":vectors,
            'label':row['label']
        })
            
    except:
        pass
    print("Done!")

vectors_df.to_pickle("./twitter_data/vectors.pkl")