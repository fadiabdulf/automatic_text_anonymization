# automatic_text_anonymization
This repository contains the code, notebooks and data related to the Journal paper "Utility-Preserving Privacy Protection of Textual Documents"

## Project structure:

├───classes\
├───data\
│&nbsp;&nbsp;&nbsp;├───hits\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;entities_generalizations4.csv &nbsp;&nbsp;&nbsp;# Generalizations of the entities from YAGO and WordNet. \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;hits.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Number of hits for the terms (collected from google). \
│&nbsp;&nbsp;&nbsp;├───NewArticles\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;wiki_Actor_XYZ.xml &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # New article from Wikipedia belong to a generalized entity. \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... \
│&nbsp;&nbsp;&nbsp;└───wiki\
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└───abstract_xml\
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;actor_1.xml &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Summary about an actor form wikipedia (if the annotation tag is 1, then the summary is manually annotated.). \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;actor_2.xml \
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... \
└───notebooks\
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01_Train.ipynb\
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02_Anonymize.ipynb\
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;03_Generalize.ipynb\
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;04_Re-identify.ipynb

## Running the code

From the notebooks directory you can find the code which origenized as following:
- Firstly, run 01_Train.ipynb notebook to install all the requirements and build the word embedding model.
- Secondly, run 02_Anonymize.ipynb notebook to do the anonymization process and calculate the evaluation metrics.
- Thirdly, run 03_Generalize.ipynb notebook to do the generalization process and calculate the evaluation metrics.
- Finally, run 04_Re-identify.ipynb notebook to perform a re-identification attack.