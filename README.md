# automatic_text_anonymization
This repository contains the code, notebooks and data related to the Journal paper "Utility-Preserving Privacy Protection of Textual Documents"

## Project structure:

├───classes\
├───data\
│   ├───hits\
|           entities_generalizations4.csv # Generalization of the entities from YAGO and WordNet. \
|           hits.txt                      # Hits of the terms from google search. \
│   ├───NewArticles\
|           wiki_Actor_XYZ.xml            # New article from Wikipedia belong to a generalized entity. \
|           ... \
│   └───wiki\
│       └───abstract_xml\
|               actor_1.xml               # Summary about an actor form wikipedia (if the annotation tag is 1, then the summary is manually annotated.). \
|               actor_2.xml \
|               ... \
└───notebooks\
│       01_Train.ipynb\
│       02_Anonymize.ipynb\
│       03_Generalize.ipynb\
│       04_Re-identify.ipynb

## Running the code

From the notebooks directory you can find the code which origenized as following:
- Firstly, run 01_Train.ipynb notebook to install all the requirements and build the word embedding model.
- Secondly, run 02_Anonymize.ipynb notebook to do the anonymization process and calculate the evaluation metrics.
- Thirdly, run 03_Generalize.ipynb notebook to do the generalization process and calculate the evaluation metrics.
- Finally, run 04_Re-identify.ipynb notebook to perform a re-identify attack.