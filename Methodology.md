# CubeML

Het ontwikkelen van een machine learning algoritme om de volgende stap van een scrambled Rubik's kubus te voorspellen, op basis van de huidige toestand ([GitHub](https://github.com/mattigoofy/CubeML)).

## Methodologie

### Dataset omschrijving

Deze [dataset](https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions) bevat een methode om een Rubik’s kubus op te lossen met de CFOP oplosmethode.

Voorbeeld uit de file:

```
3 0 2 2 0 2 4 3 4 4 0 0 5 1 3 3 1 2 0 5 1 1 2 4 5 5 1 3 4 5 5 3 3 2 2 1 0 0 5 1 4 0 0 4 5 3 2 2 1 5 3 4 4 1
D2
```

Elke file bevat 10 000 solves, met 10 files in totaal. Elke solve bestaat uit ongeveer 60 moves ([source](https://www.reddit.com/r/Cubers/comments/17mhgll/how_many_moves_should_each_step_in_cfop_take_and/)).

Elke entry bestaat uit de state van de kubus en de volgende move.

#### Kubus state

Elk nummer beschrijft een tile op de kubus. De tiles liggen in de volgorde L, U, F, D, R, B, waarbij elke face bestaat uit 9 tiles, op een kubus die georienteerd is met geel boven en groen vooraan.

Elk nummer komt overeen met een kleur: 

- Rood = 0
- Geel = 1
- Groen = 2
- Wit = 3
- Oranje = 4
- Blauw = 5

Een kubus heeft 6 faces met elk 9 tiles, dus er zijn in totaal 54 features.

#### Volgende move

De volgende move is een van deze 19 states:

```
R R' R2 L L' L2 U U' U2 B B' B2 D D' D2 F F' F2 #
```

De move \`#\` wordt gespeeld als de kubus opgelost is.

### Probleemdefinitie

Dit is een supervised multiclass classification probleem.

### Preprocessing Steps

Er is niet zoveel preprocessing nodig voor deze dataset. Elke lijn die de state beschrijft wordt afgewisseld door een lijn die de move voorstelt. De state kan opgesplitst worden in 54 features, gekoppeld aan de move. Dit zou resulteren in een pandas dataset van 55 kolommen. Er zijn in totaal 10 datasets van ongeveer 10,000 solves elk. Deze datasets zouden dan allemaal samengevoegd worden.

- Feature encoding: nummer → TILE_xy = nummer
- Dimensionality reduction:
  - Solved state weg (#)
  - Dubbele moves weg (U2 → U U)
  - Alle middelste stickers blijven altijd hetzelfde; de middelste tiles bewegen nooitimport 
- Meerdere oplossingen voor 1 state: slechts 1 houden

### Model Selection

Ons eerste idee was descision tree, omdat dit het algoritme is dat je als persoon gebruikt om een Rubik’s kubus op te lossen. Dit algoritme kan mogelijks niet goed generalizeren op nieuwe states en kan onstabiel worden, dus random forest zal een beter algoritme zijn.

Indien dit niet werk, kunnen we een neuraal netwerk gebruiken.

### Hyperparameter Tuning Strategy

Random forest heeft de parameters ([scikitlearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)):

- Number of trees (n_estimators)
- Max depth (max_depth)
- Max features (max_features)
  - Aantal features dat de tree rekening mee kan houden per split. Dit betekend dat elke tree niet alle 54 features opneemt, waardoor er meer diversiteit in de forest komt.
- Minimum samples split (min_samples_split)
  - Minimum aantal samples dat je nodig hebt voordat je een node split. Een hogere waarde kan overfitting tegen gaan.
- Minimum samples per leaf (min_samples_leaf)
  - Een hogere waarde kan zorgen voor betere generalisatie.
- Class weight (class_weight)
  - Kan gebruikt worden om om te gaan met ‘class imbalance’. In de dataset zullen de moves U en R meer voorkomen dan bijvoorbeeld B en D, omdat CFOP gemaakt is voor snelheid.

De beste hyperparameter tuning strategie zal een random search zijn, omdat een grid search te veel waarden moet bekijken.

### Scoring function

Lijst van alle mogelijke scorings: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

Omdat we met een c algoritme werken dat meer dan twee klassen bevat, kunnen alle binary scoring algoritmes niet gebruikt worden.

### Validation Strategy

Het algoritme kan moves selecteren tot het aan de \`#\` move komt (wat het einde van een solve kenmerkt), daarna valideren we of dit weldegelijk de opgeloste state is. Deze methode laat toe dat meerdere ‘oplossinspaden’ kunnen gebruikt worden om aan een opgeloste state te geraken.

Dit is geen ideale oplossing omdat het voor elke check veel tijd vergt (alle moves moeten doorlopen worden per check), maar momenteel zien we nog geen betere validatie manier.

We kunnen de standaard classificatie algoritme validatie score gebruiken, dat van een state 1 move probeert te guessen (bv. uit 100 guesses waren er 62 juist). Later kunnen we kijken naar de ROC AUC grafiek om te valideren of het model beter is dan random chance.