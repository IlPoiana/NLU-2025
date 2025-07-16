# NLU 2025 - Recap

## Table of Contents
- [NLU 2025 - Recap](#nlu-2025---recap)
  - [Table of Contents](#table-of-contents)
  - [What is NLU(?)](#what-is-nlu)
  - [1. *Natural Language*](#1-natural-language)
  - [2. *LM - Language Modelling*](#2-lm---language-modelling)
    - [Zipf's power law](#zipfs-power-law)
    - [Chain Rule and Markov assumption](#chain-rule-and-markov-assumption)
    - [N-grams](#n-grams)
    - [Train LM: Bigram and Zero-Count Problem](#train-lm-bigram-and-zero-count-problem)
    - [Train LM: Smoothing and Out-Of-Vocabulary (OOV) words](#train-lm-smoothing-and-out-of-vocabulary-oov-words)
    - [Evaluating LM](#evaluating-lm)
  - [3. *Neural Networks, Transformes \& LLM*](#3-neural-networks-transformes--llm)
  - [4. *VectorSemantics*](#4-vectorsemantics)
    - [Vector Representation](#vector-representation)
      - [Word2Vec - Static example](#word2vec---static-example)
      - [ELMO \& BERT - Context-based example](#elmo--bert---context-based-example)
    - [Evaluation of Word Embeddings](#evaluation-of-word-embeddings)
  - [5. *Part Of Speech Tagging* (POS)](#5-part-of-speech-tagging-pos)
    - [Closed vs Open Classes](#closed-vs-open-classes)
    - [Grammar basics](#grammar-basics)
      - [Universal POS-Tag](#universal-pos-tag)
  - [6 *Named Entities*](#6-named-entities)
    - [NER task](#ner-task)
    - [NER Segmentation](#ner-segmentation)
    - [NER Labelling](#ner-labelling)
    - [NER evaluation](#ner-evaluation)
  - [7 *Grammars*](#7-grammars)
    - [Parsing](#parsing)
    - [Context Free Grammars (CFG)](#context-free-grammars-cfg)
    - [Dependency Grammars](#dependency-grammars)
  - [8 *Sentence Level NLU*](#8-sentence-level-nlu)
  - [9 *Meaning*](#9-meaning)
    - [Relationships between word meanings](#relationships-between-word-meanings)
    - [Meaning representations (MR)](#meaning-representations-mr)
  - [10 *Parsing Affective States*](#10-parsing-affective-states)
    - [Lexicon](#lexicon)

This file contains the main takeaways of the NLU course. Not designed or thought to be a preparation document for the exam, but more a general knowledge and relevant facts that are worth to remeber

## What is NLU(?)

**NLU** stands for **Natural Language Understanding**

## 1. *Natural Language*

## 2. *LM - Language Modelling*
> [!NOTE]Takeaway 
> Language Models are essentially mathematical models designed to compute the probability of the next word in a sequence. Context is everything together with temporal sequences!

Are used for:
- Word or sentence completion
- Important for automatically recognising speech, where we want to have the textual transcription from audio input and to do that our language model predict the most probable sequence of words.

Why language can be modelled, in particular **statistically modelled**? because **exibits structures** and these structures can be learned or better **approximated directely from data**. It's all about computing probabilities of words given other sequences of words
### Zipf's power law
> [!IMPORTANT]
> **ZIPF'S POWER LAW**
> In the context of natural language, Zipf’s Power Law illustrates that the frequency of a word is inversely proportional to its rank/distribution in a **frequency table**. 

> [!NOTE] Takeaway
> **Zipf's power law means that words are not evenly distributed in data**, so few words will appear often (like conjuctions, determiner or common nouns and verbs) and other rarely, so in the context of probabilistic inference of words given a vocabulary this has to be taken in consideration!

### Chain Rule and Markov assumption
What are the mathematical tools that ehnance us to do language modelling? a.k.a. what are the simplest models that do probabilistic inference on words sequences?

**Chain Rule(Bayes)**: *In general, for a sequence of events x1, x2, x3, ..., xn, their joint probability P(x1, x2, x3, ..., xn) can be expressed as: P(x1) * P(x2|x1) * P(x3|x1, x2) * ... * P(xn|x1, ..., xn-1).* => A sentence, which is a words sequence, can be described as the chain rule probabilities of each subsequential word in the sentence.

> [!WARNING]
> Chain probability requires the entire sequence probabilities and the joint probabilities of each element(word)! => Really expensive
 

### N-grams
Chain Rule is not good in real applications, so we use the **Markov Assumption** which states: *"the probability of the next word depends only on a limited window of preceding words, rather than the entire history."*

Thanks to the **Markov Assumption**, we introduce the n-grams concept. Intuetively, the "n" in n-grams stands for the "window size" we consider as context for the probability computation.

> [!NOTE] Takeaway
> **N-grams** are simplifications make through the **Markov Assumption** to have a practical language model. N = 2 is similar to the **Hidden Markov Model** which has two distribution to learn rather than one. 

### Train LM: Bigram and Zero-Count Problem
> [!NOTE]Takeaway
> **N-grams suffers from Zero count and so 0 probability**, which is unrealistically modelling the natural language(for many cases, some could be zero actually)

The idea is to train models from large datasets of sentences. From these datasets, a **co-occurrence matrix** can be constructed, it tells how frequently pairs of words appear together.
From this we can computed the **raw bigram probabilities** that tells us how likely a word follows another word in the vocabulary(words but also tokens)
> [!Warning]
> Zipf's power law tells us that some words will have very low probabilities! Also some words could be just OOV and have zero probability.


### Train LM: Smoothing and Out-Of-Vocabulary (OOV) words
> [!NOTE] Takeaway
> There is no real best smoothing algorithm, but smoothing is tipically required to handle rare occurring words sequences.

To address the Zero-Count problem in N-Grams, Smoothing Algorithms comes into play:
- **Add-One(Laplace Smoothing)**: Just add one to every possible n-gram, main problem is that unbalance the probabilities of really frequent observed events(n-grams)
- **Good-Turing**: Based of **frequency-to-frequency**, is counting how many times(frequency) elements appeared certain amount of times(frequency) in my dataset. Thanks to this measure it balanced out all the frequencies, the **probability of an unseen event is** $n1\frac{1}{N}$, where n1 is the count of items that appeared only once, and N is the total number of items. This causes a drawback, **hallucinations**
- **Backoff**: If an n-gram count is 0, we consider the lower rank (n-1)-gram and operate on it to extract a non-zero count. Various methods exists, for example KneserNey uses a discount rate over the lower ranks to combine and predict the higher ones.

Another problem are OOV words that may appear in the inference or test phass, if a word has never been seen by the model, how to treat it? 

> [!TIP]
> The common usage is the introduce a \<UNK\> token for every OOV, in modern LM, tipically LLMs they have large enough vocab size, expressed in tokens, that ensures almost no OOV words.

### Evaluating LM 
Evaluating a Language Model is not straightforward, it's task dependent. There is a difference between the linguistic and statistical evaluation, a Grammar syntass and semantic will not be represented by some statistical predictive model.

For the statistical estimation, one key measure is **Perplexity** which derives from cross-entropy, and is used as measure between the generating distribution of my word sequence and the one from the LM.
- Low ppl means the model is "sure" or less perplex about its prediction, and generally means a model is good. But there is a strong assumption, **that the test data is representative of the languange** 

> [!NOTE] Takeaway
> **Perplexity** (ppl) is a quantitative measure for understanding a model predictive goodness, so its capability of predicting sequence of words given another.

> [!Warning]
> **ppl** and **WER** (Word Error Rate) are empirically correlated but not mathematically, WER is a task specific metric for Automatic Speech Recognition ppl is a predictive measure for a LM given a sequence of words.

## 3. *Neural Networks, Transformes & LLM*




## 4. *VectorSemantics*
> [!NOTE]Takeaway
> "*A word’s meaning is characterized by the company it keeps*"

A word meaning is tipically characterized by its **surrounding context**.
> [!WARNING]
> Two words can behave similarly(in terms of semantic distribution) but have different meaning!
> **Theoretical hypothesis**
> 
> Linguistic items with: 
> 1. *similar distributions* have
> 2. *related meanings* (behave similarly)
> 3. in *similar contexts*. 

### Vector Representation
Words can be mapped into a geometric space. In this conceptual space, **each word is represented by a unique point**(a vector pointing to that point). The proximity and relationships between these points can then be used to understand the similarity or relatedness of words.

- **One Hot**
  - Naive,simple but hard to use
- **Frequency Based**
  - Cross-occurency matrix transforming each row (e.g. verb) in to a vector of dimension equal to the number of co-occurences between each of the paired tokens(e.g nouns)
- **Information Retrieval**
  - We follow the information retrieval theory of extracting relevant words in documents by "normalizing" their representation (boosting in the case of tf-idf) based on the frequency of appearance in documents of some vocabulary tokens.  
  - For example **tf-idf**
  - **PPMI**
- **Unsupervised approaches(Simple)**
  - **Clustering**
- **Neural Embeddings**
  - **Static:** each word is represented **as a fixed, pre-trained vector** of real numbers. These vectors are generated based on the context of the word in a large corpus of text data and **remain the same for each instance** of the word across all contexts.
  - **Context-based:**
  -  fek

#### Word2Vec - Static example

Is a **self-supervised**, **static word embedding model**, which leverages the prediction of next word(token) task to train a model **which can create a vectorization of my vocabulary.**

> [!NOTE]Takeaway
> The idea here is to **predict rather than count**

> [!WARNING]  
> This is trained as a classifier, so it suffers from the biases that comes from the data!

The result of this method was showing a more semantically structured embedding space of words in the vocabulary.

> [!Tip]
> Similarity measures compared
> - **Raw dot product** suffer from the exploding score depending from the vector length 
> - **Cosine similarity** normalize the raw dot product and get a more accurate and reliable similarity score

#### ELMO & BERT - Context-based example

- **ELMO**: uses a bidirectional LSTM to encode each word, LSTM uses the all sequence to encode each token
- **BERT**: Bidirectional Encoder Representations from Transformers, uses transformer based architecture to encode the sequence.

> [!Note] Takeaway
> *Semantic similarity is only a weak reflection of actual meaning*

Bert family of words embedders based their "knowledge" on similarity measure, this doesn't entail the fact that they code **words meaning** 

### Evaluation of Word Embeddings

Different kind of evaluation, which each adresses a specific side of NLU.

- **Intrinsic**: Embeddings versus human evaluation/judgements. Like word similarity ratings.
- **Extrinsic**: Evaluate the embeddings on downstream NLP applications, like Name Entity Recongnition, Question Answering, Semantic Role Labelling (shallow semantic parsing)(Slot Filling),



## 5. *Part Of Speech Tagging* (POS)
> [!Note] Takeaway
> Part Of Speech tagging is the grammatical "syntactical" classifications of words Is the baseline for more complex shallow parsing tasks, like Named Entity Recognition, this task is considered a sequence labelling task.


Part of Speech (POS) tagging is the process of **assigning the correct grammatical category (POS tag) to each word in a given text.**
This fall under the sequence labeling task.

### Closed vs Open Classes
- **Closed classes** are generally comprised of function words such as prepositions and pronouns.
- **Open Classes** In contrast, open classes have a virtually infinite number of members. Like nouns or verbs

### Grammar basics

> [!Note] Takeaway
> - **High frequency**: Pronouns, Conjuctions, some verbs(to be, have, can)
> - **Low Frequency**: Verbs(to scout, to treat), nouns

- **Pronouns**: Pronouns are vital for the economics of communication. They act as a shorthand for referring to entities or events. 
  - Ex: I,me, mine, yours(possessive), who, what
- **Nouns**: Nouns are words that represent people, places, or things.
- **Verbs**: Verbs are words that denote actions or processes. Verbs often undergo *inflection* depending on tense, person, and number.
- **Adverbs**: Adverbs constitute a highly heterogeneous class of words that primarily modify verbs, other adverbs, and verb phrases.
  - Ex: -ly words(slowly, extremely...)
- **Conjuctions**: Conjunctions are words that connect two sentences, phrases, or clauses.
  - Ex: and, or, so

> [!Tip]
> *Inflection* is the process by which words change form to encode grammatical information (rather than to derive new words).

#### Universal POS-Tag
Universal POS-Tag Set represents a simplified and unified set of part-of-speech tags, that was proposed for the standardization across corpora and languages. 
The number of defined tags varies from 12 ([Petrov et al/Google/NLTK](https://github.com/slavpetrov/universal-pos-tags)) to 17 ([Universal Dependencies/spaCy](https://universaldependencies.org/u/pos/index.html), in *Italics*).



| Tag  | Meaning | English Examples |
|:-----|:--------|:-----------------|
| __Open Class__ |||
| NOUN | noun (common and proper) | year, home, costs, time, Africa
| VERB | verb (all tenses and modes) | is, say, told, given, playing, would
| ADJ  | adjective           | new, good, high, special, big, local
| ADV  | adverb              | really, already, still, early, now
| *PROPN* | proper noun (split from NOUN) | Africa
| *INTJ*  | interjection (split from X) | oh, ouch
| __Closed Class__ |||
| DET  | determiner, article | the, a, some, most, every, no, which
| PRON | pronoun             | he, their, her, its, my, I, us
| ADP  | adposition	(prepositions and postpositions) | on, of, at, with, by, into, under
| NUM  | numeral             | twenty-four, fourth, 1991, 14:24
| PRT (*PART*) | particles or other function words | at, on, out, over per, that, up, with
| CONJ | conjunction         | and, or, but, if, while, although
| *AUX* | auxiliary (split from VERB) | have, is, should
| *CCONJ*  | coordinating conjunction (splits CONJ) | or, and
| *SCONJ*  | subordinating conjunction (splits CONJ) | if, while
| __Other__ |||
| .    | punctuation marks   | . , ; !
| X    | other               | foreign words, typos, abbreviations: ersatz, esprit, dunno, gr8, univeristy
| *SYM* | symbols (split from X) | $, :) 

## 6 *Named Entities*
*"NLU maps a natural language sentence in a semantic representation ( e.g. database semantics ) that includes entities (e.g. flight, airport, meals) and their relations ."*

Entities and their relations are semantically relevant for understanding different aspects of sentences. There is no universal agreement of how to define them.

In NLU we restrict our analysis to **named entities**, which are entities "named" or with a label that represent some form of classification.

### NER task

> [!NOTE] Takeaway
> Named Entity Recognition (NER) it's a NLU task which is all about detect and name(classifing those, assignign a label) words, sometimes grouped together, in text sequences related to the "entities" or things we know about, literally "we can name of". Often is a straight follow up of POS tagging.

The Named Entity Recognition (NER) task involves finding Named Entities and their specific types within a given text. This process typically has two main components:
1. **Named Entity Recognition Segmentation:** This step focuses on identifying the names or spans of text that represent entities.
2. **Named Entity Recognition Labeling:** Once segmented, this step involves classifying each identified NE into its appropriate category. For example, classifying ‘Washington’ as a Location, ‘Andrew Wilkie’ as a Person, or ‘Labor government’ as an Organization.
  
It doesn't exist a single set of NE labels, many have been proposed and the choice is tied to the situation. There are also subcategories of NER, like **Slot Filling** which aims to named only entities relevant for certain domains and not any text.

### NER Segmentation
Again, segmentation is fundamental for having the words/tokens indicating which are entities to name and which are just words, so again here the importance of POS tagging, remember the noun-verb situation where a word can be noun and verb, so in one case entity to named and in the other not.

- *IO*: I- tokens for inside, marking the inside entity token, O- the opposite, outside
- *BIO*: B- begin tokens added to IO schema, permits to represent multi token entities(which is a more realistic case) 

> [!Warning]
> NE sometimes are nested, which means that a NE can contain one or more NE or nested NE. So Hierachical segmentation techniques must be considered, especially in Human-Machine dialogue.


### NER Labelling 
Labelling is an important part, to do so you have first to understand what type of NE suits better your problem:

- **"Closed" NE**: like spaCy have 18 fixed ones, or other have less or more. This permits smaller models and easier evaluation also.
- **"Open" NE**: are NE classes that contains many entities categories and are not limited in the number. This permits to named more entities also the ones rare or domain specific that can be desired to solve field specific tasks.

### NER evaluation
Evaluating NE comes with some problems, but tipically for basic NE you can rely on **Precision & Recall** metric.

> [!Tip]
> f1 score is a composition of Precision and Recall scores. Both consider TP count, but Precision consider the TN and Recall the FN.

The common problems are:
1. Non-english like languages
2. Different domain and sources
3. Segmentation errors

This can't be done efficiently for Open NE unless a big amount of data is used to remove defects and isolate cases. 

## 7 *Grammars*
> [!NOTE] Takeaway
> It doesn't exist a strong, universal semantical and syntactical parser for all languages, but many powerful ones have been proposed. The role of understanding and getting structures and relations in natural language is key for solving many tasks like the meaning of a phrase. CFG are limited respect to Dependency Grammars. 

Natural Language is ambiguos, so a deep understanding of syntactic and semantic structure is crucial. 

### Parsing 
Parsing in general is the process of taking an input string (a sentence) and returning **zero or more plausible parse trees for that string**, so to uncovers its undrelying structure and meaning.

- **Syntactic parsing**: Involves understanding how words group together to form meaningful units, so the sentence division of subject object ecc. ecc.
- **Semantic parsing**: Semantic Parsing aims to extract the meaning from a sentence, often by answering questions such as “Who did What to Whom”.

What are the tools for parsing?
1. CFG
2. Dependecy Grammars
3. SpaCy
4. LLM
### Context Free Grammars (CFG)
>[!NOTE] Takeaway
> CFG (sometimes referred as belonging to Constituency grammars) is a powerful tool for parsing (analysis and checking) and generating syntactical correct sentences. They lack basically from the semantic meaning when constructing sequences, and to solve this a lot of complexity appear typically when modelling one.

Context Free Grammars, as the name suggest are grammars, set of rules that applied recursively which generates any valid sentence for a language. They have an expressive power, meaning they can and cannot create certain sequences/sentences.
The main characteristic is that they are **context free**, so they don't use word meaning for generating sequences, but only sets of symbols and rules to work.

Four rules defined a CFG:
1. Terminal symbols
2. Non-terminal symbols
3. Productions (generative rules, transition)
4. Starting point 

CFG capture **constituency** and **ordering**:
- **Constituency**: How words groups into costituents
- **Ordering**: What is the ordering of the words in the sentences

> [!Warning]
> CFG for they independence from the context, or meaning between possible word generation might lead to **over-generation**, or creating semantically non-sensed phrases

Relevant problems with CFG are:
1. **Agreement**: sub-rules for Noun Phrases and Verb Phrases are not specified, so there is the need to refactor the rules and add much more complexity to the CFG to handle this, but it's possible.
2. **Subcategorization**: Verbs are not created equally, so there is a Need of sub-categorization to solve this issue, also here adding more complexity
3. **Movement**: sometimes the verbs appear far from the expected position, so long distances dependencies are hard to capture for CFG
4. **Scalability**: Putting all together, Context Free Grammars struggle to scale when referencing Natural Language.

### Dependency Grammars

> [!NOTE]
> Dependency Grammars, and in particular parsing is excellent for extracting semantic relations in sentences. Especially in cases where the relations are "buried" for constituency parses instead here the relations are more explicit

In dependency grammars we reason on **heads** and **dependent**, where the head have a **binary asymmetric relation** with the dependent.

Relations types are:
- **Nominal subjects**: nominal subject
- **Obj**: direct object
- **Det**: determiner
- **Compound**: multiword lexeme
- and much mrore

EXAMPLE:
  [I prefer a morning flight] => prefer:HEAD -> [I:NSUBJ, [flight:OBJ -> a:det, morning:compound] ]

In dependency parsers the arrows, or relations, are typed with grammatical relations.
This creates dependency graphs, that are tree structured, and each element of the graph is reached by the head at some level and have exactly only one relation pointing to him.

**Projectivity in a dependency graph** when there are no crossing arcs. This if present gurantees some algorithms to work, tipically fast ones.


## 8 *Sentence Level NLU*
> [!NOTE] Takeaway
> NLU in practice is Sequence Labelling which models sentence meaning. To do so two things have to be done, **segmentation** and **labelling**. The first is alla about grouping words that mapped to the same concept, the second is assigning a label/category to this concepts. Examples are slot filling and intent classification tasks.

> [!Tip]
> Meaning is conceptualized as an abstract representationi of explicit signals. They could be text, speech or movements.

Now after parsing the sentence and understood its key components and relations, we actually have to NLU, so **understand what is the argument of the conversation/sentences**. This translates into **Intent classification** for example.

One kind of NLU, that we have already seen, is sequence modelling/labelling. Like Slot Filling is not only assigning tags to each utterance but also to assign to those what "is" that utterance **in the specific context of the conversation**.

Modern NLU focused on **Data-Driven Learning**, like ML or DL, where huge amount of labelled data (or unlabelled in unsupervised learning) is required to learn their distribution and train models to learn the relations of natural lanugage.

> [!Warning]
> ML and DL models, such as RNN, LSTM and Tranformers doesn't have an internal representation of "meaning", they learn distributions by similarity scores (like Cross-Entropy) directly from data, without any kind of rules or previous structural knwoledge (like CFG or Dependency Grammars).
 
> [!Warning]
> The concept of intents often suffers from **fuzzy definitions and is prone to issues like over-generalisation or under-generalisation.** Furthermore, there is no universal lexicon for intents, which hinders reusability and extendability across different systems.

## 9 *Meaning*
>[!NOTE]
> *"An abstract representation of explicit signals such as speech, text, gestures"*

Meaning in phrases goes with NLU and has many applications, it goes from "simple word" level meaning (Fire Truck, what does it mean?) like understanding what is happening in a sentence (The big star is behind the little star).

This has been studied in three different perspectives:
1. **Lexical Semantics**: Meaning of individual words (fire)
2. **Sentential Semantics**: Meaning of sentences or grouped words (fire truck)
3. **Discourse**: Meaning for debates, discourses and complex and long texts (Journal page)

- *Lexeme*: a unit of meaning denoted by a base form.  
- *Lemmatization*: process of taking a word and map it to its lexeme base form
  -**Found** can be a wordform of "find" or just "found".
- *Stemming*: is a simpler, more rule-based approach to reducing wordforms to a base or root form, primarily by stripping off suffixes. For example, “going” would be stemmed by removing “-ing” to yield “go”. Unlike lemmatization, stemming does not necessarily produce a valid dictionary word and generally does not account for different word senses or the broader linguistic context.

### Relationships between word meanings

Below is a summary of the five core semantic relationships between word meanings:

| Relationship             | Description                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------- |
| **Homonymy**             | Same form (spelling/pronunciation), unrelated meanings (e.g. bat: animal vs. club). |
| **Polysemy**             | One word, multiple related senses (e.g. bank: institution vs. building).            |
| **Synonymy**             | Different words, near‐identical meaning (e.g. car/automobile), rarely perfect.      |
| **Antonymy**             | Opposites on a scale yet semantically related (e.g. hot/cold, up/down).             |
| **Hypernymy / Hyponymy** | “Is‐a” hierarchy: hyponym more specific, hypernym more general (e.g. dog/canid).    |

> [!Warning]
> This pose the need of NLU as it is structured today, and more importantly the fact that **Natural Language is ambiguos** typically context dependent and there is no general model, rule, structure to 100% surely disambiguate what is said. Engage with the user, see words connections, semantically speaking and by a meaning perspective, knowing who is talking or reading are all factors that counts in NLU.

### Meaning representations (MR)

A high-quality meaning representation (MR) for computational use should satisfy six core desiderata:

1. **Verifiability**: link each MR unambiguously to a knowledge base so its truth can be checked and explained (e.g. IS-A(Kazakhstan, Country)).
2. **Completeness**: capture every plausible interpretation of an ambiguous input, leaving disambiguation to later modules.
3. **Robustness**: map varied surface expressions of the same idea onto a single canonical form (e.g. both “Charrito has gluten-free dishes” and “You can get gluten free at Charrito” → SERVES(Charrito, GlutenFreeFood)).
4. **Expressiveness**: use a formalism rich enough to represent all intended linguistic constructions.
5. **Inference**: support logical reasoning (e.g. allow variables: SERVES(x, GlutenFreeFood) to denote “some restaurant”).
6. **Uncertainty** : attach confidence scores at the MR and/or sub-component level (e.g. SERVES[0.85](Charrito[0.8], GlutenFreeFood[0.5])) to reflect recognition or semantic ambiguity.

Different approaches exists, like: **FOL** and **semantic networks**.

## 10 *Parsing Affective States*

### Lexicon

> [!NOTE] Takeaway
> 
