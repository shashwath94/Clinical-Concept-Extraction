######################################################################
#  CliNER - model.py                                                 #
#                                                                    #
#  Willie Boag                                                       #
#                                                                    #
#  Purpose: Define the model for clinical concept extraction.        #
######################################################################



from sklearn.feature_extraction  import DictVectorizer

import crf
from documents import labels as tag2id, id2tag
from tools import flatten, save_list_structure, reconstruct_list



class Model:

    def __init__(self):
        """
        Model::__init__()

        Instantiate a Model object.
        """
        self._is_trained     = None
        self._clf            = None
        self._dvect          = None
        self._training_files = None


    def fit_from_documents(self, documents):
        # Extract formatted data
        tokenized_sents  = flatten([d.getTokenizedSentences() for d in documents])
        labels           = flatten([d.getTokenLabels()        for d in documents])

        # train classifier
        dvect, clf = train_classifier(tokenized_sents, labels)

        self._is_trained = True
        self._dvect = dvect
        self._clf   = clf


    def predict(self, document):
        # Extract formatted data
        tokenized_sents  = document.getTokenizedSentences()

        # Predict labels for prose
        num_pred = predict_classifier(tokenized_sents         ,
                                      dvect    = self._dvect  ,
                                      clf      = self._clf    )

        iob_pred = [ [id2tag[p] for p in seq] for seq in num_pred ]

        return iob_pred



def make_feature(ind):
    #return {ind:1}
    return {(ind,val):1 for val in range(20)}



def extract_features(sent):
    fseq = [{'dummy':1} for _ in sent]

    for i in range(len(sent)):
        #print sent[i]
        for j in range(20):
            fseq[i][('word',sent[i],j)] = 1
        #print fseq[i]
        #print 
    #exit()
    return fseq



def train_classifier(tokenized_sents, iob_nested_labels):
    # Must have data to train on
    if len(tokenized_sents) == 0:
        raise Exception('Training must have training examples')

    print '\tvectorizing words'

    # vectorize tokenized sentences
    text_features = []
    #for sent in tokenized_sents:
    for lineno,sent in enumerate(tokenized_sents):
        fseq = extract_features(sent)
        text_features.append(fseq)

    # Save list structure to reconstruct after vectorization
    offsets = save_list_structure(text_features)

    # Vectorize features
    dvect = DictVectorizer()
    X_feats = dvect.fit_transform( flatten(text_features) )

    # nested features
    X_feats = reconstruct_list( list(X_feats) , offsets)

    # vectorize IOB labels
    Y_labels = [ [tag2id[y] for y in y_seq] for y_seq in iob_nested_labels ]

    print '\ttraining classifiers'

    # train classifier
    clf = crf.train(X_feats, Y_labels)

    '''
    pred = crf.predict(clf, X_feats)
    assert len(Y_labels) == len(pred)
    for i in range(len(Y_labels)):
        print Y_labels[i]
        print pred[i]
        print
    exit()
    '''

    return dvect, clf



def predict_classifier(tokenized_sents, dvect, clf):
    # If nothing to predict, skip actual prediction
    if len(tokenized_sents) == 0:
        print '\tnothing to predict '
        return []

    print '\tvectorizing words ' 

    # vectorize tokenized sentences
    text_features = []
    for sent in tokenized_sents:
        fseq = extract_features(sent)
        text_features.append(fseq)

    # Save list structure to reconstruct after vectorization
    offsets = save_list_structure(text_features)

    # Vectorize features
    X_feats = dvect.transform( flatten(text_features) )

    # nested features
    X_feats = reconstruct_list( list(X_feats) , offsets)

    # Predict labels
    pred = crf.predict(clf, X_feats)

    # Format labels from output
    return pred



