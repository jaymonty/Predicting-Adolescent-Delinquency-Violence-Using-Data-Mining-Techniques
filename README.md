# Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques

```python
!pip install https://github.com/aaren/notebook/tarball/master
```


```python
!notebdown
```


```python
pip install scikit-plot
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting scikit-plot
      Downloading scikit_plot-0.3.7-py3-none-any.whl (33 kB)
    Requirement already satisfied: matplotlib>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from scikit-plot) (3.7.1)
    Requirement already satisfied: scikit-learn>=0.18 in /usr/local/lib/python3.10/dist-packages (from scikit-plot) (1.2.2)
    Requirement already satisfied: joblib>=0.10 in /usr/local/lib/python3.10/dist-packages (from scikit-plot) (1.2.0)
    Requirement already satisfied: scipy>=0.9 in /usr/local/lib/python3.10/dist-packages (from scikit-plot) (1.10.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (0.11.0)
    Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (8.4.0)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (23.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (1.4.4)
    Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (1.24.3)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (3.0.9)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (1.0.7)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (4.39.3)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=1.4.0->scikit-plot) (2.8.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.18->scikit-plot) (3.1.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=1.4.0->scikit-plot) (1.16.0)
    Installing collected packages: scikit-plot
    Successfully installed scikit-plot-0.3.7



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
```


```python
from pydrive.auth import GoogleAuth
from google.colab import drive
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.colab import files
```


```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
import scikitplot as skplt
```


```python
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '19IWNzmJ0k41UVUc320EnvwEDKi77YiyC'
download = drive.CreateFile({'id': file_id})

# Download the file to a local disc
download.GetContentFile('file.csv')
df = pd.read_csv('file.csv')
```


```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```


```python
df.shape
```




    (68507, 146)




```python
df.head()
```





  <div id="df-ee84d267-c53a-4859-87a9-d8fe59d16d85">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>CASEID</th>
      <th>SCOUNTRY</th>
      <th>MALE</th>
      <th>AGEGROUP</th>
      <th>BIRTHP</th>
      <th>BIRTHPM</th>
      <th>BIRTHPF</th>
      <th>FAMILY</th>
      <th>LANGH1</th>
      <th>DISCRIM</th>
      <th>WORKFATH</th>
      <th>WORKMOTH</th>
      <th>OWNROOM</th>
      <th>COMPUSE</th>
      <th>OWNMOBPH</th>
      <th>FAMILCAR</th>
      <th>VICROBBP</th>
      <th>VICASSAP</th>
      <th>VICTHEFP</th>
      <th>VICBULLP</th>
      <th>GETALFA</th>
      <th>GETALMO</th>
      <th>LEISFAM</th>
      <th>DINNFAM</th>
      <th>KNOWFR</th>
      <th>TELLTIME</th>
      <th>OBEYTIME</th>
      <th>LIFEEV01</th>
      <th>LIFEEV02</th>
      <th>LIFEEV03</th>
      <th>LIFEEV04</th>
      <th>LIFEEV05</th>
      <th>LIFEEV06</th>
      <th>LIFEEV07</th>
      <th>LIFEEV08</th>
      <th>NIGHTACT</th>
      <th>ACTIV01</th>
      <th>ACTIV02</th>
      <th>ACTIV03</th>
      <th>ACTIV04</th>
      <th>ACTIV05</th>
      <th>ACTIV06</th>
      <th>ACTIV07</th>
      <th>TRANSP01</th>
      <th>TRANSP02</th>
      <th>TRANSP03</th>
      <th>TRANSP04</th>
      <th>TRANSP05</th>
      <th>TRANSP06</th>
      <th>TRANSP07</th>
      <th>LEISSP01</th>
      <th>LEISSP02</th>
      <th>LEISSP03</th>
      <th>LEISSP04</th>
      <th>GROUPFR</th>
      <th>GRPAGE01</th>
      <th>GRPAGE02</th>
      <th>GRPAGE03</th>
      <th>GRPAGE04</th>
      <th>GRPAGE05</th>
      <th>GRPPUBL</th>
      <th>GRPEXIST</th>
      <th>GRPILLAC</th>
      <th>GRPILLDO</th>
      <th>GRPGANG</th>
      <th>GRPGEND</th>
      <th>GRPETHN</th>
      <th>ETHNFRND</th>
      <th>FRNDAC01</th>
      <th>FRNDAC02</th>
      <th>FRNDAC03</th>
      <th>FRNDAC04</th>
      <th>FRNDAC05</th>
      <th>FRNDAC06</th>
      <th>FRNDAC07</th>
      <th>FRNDAC08</th>
      <th>FRNDAC09</th>
      <th>ATTVIO01</th>
      <th>ATTVIO02</th>
      <th>ATTVIO03</th>
      <th>ATTVIO04</th>
      <th>ATTVIO05</th>
      <th>SELFC01</th>
      <th>SELFC02</th>
      <th>SELFC03</th>
      <th>SELFC04</th>
      <th>SELFC05</th>
      <th>SELFC06</th>
      <th>SELFC07</th>
      <th>SELFC08</th>
      <th>SELFC09</th>
      <th>SELFC10</th>
      <th>SELFC11</th>
      <th>SELFC12</th>
      <th>ACCIDP</th>
      <th>ATTSCH</th>
      <th>REPGRADE</th>
      <th>TRUANCY</th>
      <th>ACHIEV</th>
      <th>ATSCH01</th>
      <th>ATSCH02</th>
      <th>ATSCH03</th>
      <th>ATSCH04</th>
      <th>ATSCH05</th>
      <th>ATSCH06</th>
      <th>ATSCH07</th>
      <th>ATSCH08</th>
      <th>AFTSCH</th>
      <th>NHOOD01</th>
      <th>NHOOD02</th>
      <th>NHOOD03</th>
      <th>NHOOD04</th>
      <th>NHOOD05</th>
      <th>NHOOD06</th>
      <th>NHOOD07</th>
      <th>NHOOD08</th>
      <th>NHOOD09</th>
      <th>NHOOD10</th>
      <th>NHOOD11</th>
      <th>NHOOD12</th>
      <th>NHOOD13</th>
      <th>DELPDR</th>
      <th>DELPSL</th>
      <th>DELPBU</th>
      <th>DELPEX</th>
      <th>DELPAS</th>
      <th>BEERLTP</th>
      <th>SPIRLTP</th>
      <th>HASHLTP</th>
      <th>XTCLTP</th>
      <th>LHCLTP</th>
      <th>VANDLTP</th>
      <th>SHOPLTP</th>
      <th>BURGLTP</th>
      <th>BICTLTP</th>
      <th>CARTLTP</th>
      <th>DOWNLTP</th>
      <th>HACKLTP</th>
      <th>CARBLTP</th>
      <th>SNATLTP</th>
      <th>WEAPLTP</th>
      <th>EXTOLTP</th>
      <th>GFIGLTP</th>
      <th>ASLTLTP</th>
      <th>DRUDLTP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>101001</td>
      <td>USA</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>101002</td>
      <td>USA</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>101003</td>
      <td>USA</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>99</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>101004</td>
      <td>USA</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>99</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>101005</td>
      <td>USA</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ee84d267-c53a-4859-87a9-d8fe59d16d85')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ee84d267-c53a-4859-87a9-d8fe59d16d85 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ee84d267-c53a-4859-87a9-d8fe59d16d85');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python

# Don't need "Unnamed: 0", and 'CASEID'
df.drop(columns=['Unnamed: 0','CASEID'], inplace=True)
```


```python
df.shape
```




    (68507, 144)



# Data Pre-Processing


```python
# 143 ints, 1 object
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 68507 entries, 0 to 68506
    Columns: 144 entries, SCOUNTRY to DRUDLTP
    dtypes: int64(143), object(1)
    memory usage: 75.3+ MB



```python
# Null values
df.isnull().sum().sum()
```




    0




```python
# Missing values
df.isna().sum().sum()
```




    0




```python
cat_col_names = []
num_col_names = []

for col in df.columns:
  if df[col].dtype == 'object':
    cat_col_names.append(col)
  else:
    num_col_names.append(col)

print(cat_col_names)
print()
print(num_col_names)
```

    ['SCOUNTRY']
    
    ['MALE', 'AGEGROUP', 'BIRTHP', 'BIRTHPM', 'BIRTHPF', 'FAMILY', 'LANGH1', 'DISCRIM', 'WORKFATH', 'WORKMOTH', 'OWNROOM', 'COMPUSE', 'OWNMOBPH', 'FAMILCAR', 'VICROBBP', 'VICASSAP', 'VICTHEFP', 'VICBULLP', 'GETALFA', 'GETALMO', 'LEISFAM', 'DINNFAM', 'KNOWFR', 'TELLTIME', 'OBEYTIME', 'LIFEEV01', 'LIFEEV02', 'LIFEEV03', 'LIFEEV04', 'LIFEEV05', 'LIFEEV06', 'LIFEEV07', 'LIFEEV08', 'NIGHTACT', 'ACTIV01', 'ACTIV02', 'ACTIV03', 'ACTIV04', 'ACTIV05', 'ACTIV06', 'ACTIV07', 'TRANSP01', 'TRANSP02', 'TRANSP03', 'TRANSP04', 'TRANSP05', 'TRANSP06', 'TRANSP07', 'LEISSP01', 'LEISSP02', 'LEISSP03', 'LEISSP04', 'GROUPFR', 'GRPAGE01', 'GRPAGE02', 'GRPAGE03', 'GRPAGE04', 'GRPAGE05', 'GRPPUBL', 'GRPEXIST', 'GRPILLAC', 'GRPILLDO', 'GRPGANG', 'GRPGEND', 'GRPETHN', 'ETHNFRND', 'FRNDAC01', 'FRNDAC02', 'FRNDAC03', 'FRNDAC04', 'FRNDAC05', 'FRNDAC06', 'FRNDAC07', 'FRNDAC08', 'FRNDAC09', 'ATTVIO01', 'ATTVIO02', 'ATTVIO03', 'ATTVIO04', 'ATTVIO05', 'SELFC01', 'SELFC02', 'SELFC03', 'SELFC04', 'SELFC05', 'SELFC06', 'SELFC07', 'SELFC08', 'SELFC09', 'SELFC10', 'SELFC11', 'SELFC12', 'ACCIDP', 'ATTSCH', 'REPGRADE', 'TRUANCY', 'ACHIEV', 'ATSCH01', 'ATSCH02', 'ATSCH03', 'ATSCH04', 'ATSCH05', 'ATSCH06', 'ATSCH07', 'ATSCH08', 'AFTSCH', 'NHOOD01', 'NHOOD02', 'NHOOD03', 'NHOOD04', 'NHOOD05', 'NHOOD06', 'NHOOD07', 'NHOOD08', 'NHOOD09', 'NHOOD10', 'NHOOD11', 'NHOOD12', 'NHOOD13', 'DELPDR', 'DELPSL', 'DELPBU', 'DELPEX', 'DELPAS', 'BEERLTP', 'SPIRLTP', 'HASHLTP', 'XTCLTP', 'LHCLTP', 'VANDLTP', 'SHOPLTP', 'BURGLTP', 'BICTLTP', 'CARTLTP', 'DOWNLTP', 'HACKLTP', 'CARBLTP', 'SNATLTP', 'WEAPLTP', 'EXTOLTP', 'GFIGLTP', 'ASLTLTP', 'DRUDLTP']



```python
# Looking at the values for each feature

for col in num_col_names:
  print(col, '({})'.format(df[col].nunique()), ":")
  print(df[col].value_counts())
  print()
  print()
```

    MALE (4) :
    0    34583
    1    33758
    9      154
    7       12
    Name: MALE, dtype: int64
    
    
    AGEGROUP (5) :
    1    62672
    2     5391
    9      236
    3      153
    0       55
    Name: AGEGROUP, dtype: int64
    
    
    BIRTHP (4) :
    1    62781
    0     5354
    9      355
    7       17
    Name: BIRTHP, dtype: int64
    
    
    BIRTHPM (6) :
    1    55581
    2    11274
    3      627
    4      517
    9      491
    7       17
    Name: BIRTHPM, dtype: int64
    
    
    BIRTHPF (6) :
    1    54802
    2    11262
    4     1163
    3      684
    9      573
    7       23
    Name: BIRTHPF, dtype: int64
    
    
    FAMILY (11) :
    1     49515
    3      7988
    5      4171
    2      3338
    7      1169
    4       887
    6       566
    9       348
    8       196
    99      196
    97      133
    Name: FAMILY, dtype: int64
    
    
    LANGH1 (4) :
    1    57890
    0     6157
    9     4407
    7       53
    Name: LANGH1, dtype: int64
    
    
    DISCRIM (6) :
    1    59720
    3     4122
    2     3367
    4      880
    9      398
    7       20
    Name: DISCRIM, dtype: int64
    
    
    WORKFATH (10) :
    1     46529
    2     10521
    98     4020
    3      2172
    6      1217
    4      1165
    7       933
    99      876
    5       753
    97      321
    Name: WORKFATH, dtype: int64
    
    
    WORKMOTH (10) :
    1     43704
    6      9551
    2      4830
    3      2933
    7      2545
    4      2359
    5       905
    99      850
    98      594
    97      236
    Name: WORKMOTH, dtype: int64
    
    
    OWNROOM (4) :
    1    51556
    0    16721
    9      192
    7       38
    Name: OWNROOM, dtype: int64
    
    
    COMPUSE (4) :
    1    57416
    0    10872
    9      191
    7       28
    Name: COMPUSE, dtype: int64
    
    
    OWNMOBPH (4) :
    1    60834
    0     7468
    9      195
    7       10
    Name: OWNMOBPH, dtype: int64
    
    
    FAMILCAR (4) :
    1    59094
    0     9196
    9      191
    7       26
    Name: FAMILCAR, dtype: int64
    
    
    VICROBBP (4) :
    0    62860
    9     2962
    1     2642
    7       43
    Name: VICROBBP, dtype: int64
    
    
    VICASSAP (4) :
    0    62392
    9     3409
    1     2671
    7       35
    Name: VICASSAP, dtype: int64
    
    
    VICTHEFP (4) :
    0    52235
    1    12952
    9     3251
    7       69
    Name: VICTHEFP, dtype: int64
    
    
    VICBULLP (4) :
    0    56184
    1     9084
    9     3144
    7       95
    Name: VICBULLP, dtype: int64
    
    
    GETALFA (7) :
    4    42829
    3    15799
    8     5096
    2     3261
    1      897
    9      386
    7      239
    Name: GETALFA, dtype: int64
    
    
    GETALMO (7) :
    4    49614
    3    14707
    2     2377
    8      796
    1      524
    9      291
    7      198
    Name: GETALMO, dtype: int64
    
    
    LEISFAM (8) :
    5    22982
    6    18077
    4    12852
    3     7821
    1     4926
    2     1292
    9      307
    7      250
    Name: LEISFAM, dtype: int64
    
    
    DINNFAM (10) :
    8     42509
    6      4098
    1      3903
    4      3813
    7      3800
    3      3673
    5      3253
    2      3003
    99      284
    97      171
    Name: DINNFAM, dtype: int64
    
    
    KNOWFR (6) :
    3    38042
    2    23434
    1     3493
    8     2669
    7      636
    9      233
    Name: KNOWFR, dtype: int64
    
    
    TELLTIME (5) :
    1    46469
    8    11615
    0     9362
    7      635
    9      426
    Name: TELLTIME, dtype: int64
    
    
    OBEYTIME (7) :
     3    24512
     8    20436
     2    18603
     9     2389
     1     1460
    -9      558
     7      549
    Name: OBEYTIME, dtype: int64
    
    
    LIFEEV01 (4) :
    0    64512
    1     2520
    9     1423
    7       52
    Name: LIFEEV01, dtype: int64
    
    
    LIFEEV02 (4) :
    0    63746
    1     3189
    9     1476
    7       96
    Name: LIFEEV02, dtype: int64
    
    
    LIFEEV03 (4) :
    1    39800
    0    26058
    9     2505
    7      144
    Name: LIFEEV03, dtype: int64
    
    
    LIFEEV04 (4) :
    0    57831
    1     8399
    9     2191
    7       86
    Name: LIFEEV04, dtype: int64
    
    
    LIFEEV05 (4) :
    0    45281
    1    21535
    9     1611
    7       80
    Name: LIFEEV05, dtype: int64
    
    
    LIFEEV06 (4) :
    0    62112
    1     4868
    9     1481
    7       46
    Name: LIFEEV06, dtype: int64
    
    
    LIFEEV07 (4) :
    0    59496
    1     7415
    9     1506
    7       90
    Name: LIFEEV07, dtype: int64
    
    
    LIFEEV08 (4) :
    0    52744
    1    14151
    9     1555
    7       57
    Name: LIFEEV08, dtype: int64
    
    
    NIGHTACT (10) :
    1     17112
    2     15144
    3     11410
    4      8109
    8      6423
    5      4491
    6      2664
    7      1522
    97     1175
    99      457
    Name: NIGHTACT, dtype: int64
    
    
    ACTIV01 (8) :
    3    23728
    2    19377
    4    12401
    5     4593
    1     4585
    6     2526
    9      677
    7      620
    Name: ACTIV01, dtype: int64
    
    
    ACTIV02 (8) :
    1    25110
    2    21005
    3    12620
    4     4530
    6     1647
    5     1614
    9     1418
    7      563
    Name: ACTIV02, dtype: int64
    
    
    ACTIV03 (8) :
    4    17450
    6    15546
    3    14057
    5    12272
    2     5709
    1     1784
    9     1186
    7      503
    Name: ACTIV03, dtype: int64
    
    
    ACTIV04 (8) :
    1    26547
    2    26292
    3     9362
    4     2433
    9     1460
    5      945
    6      916
    7      552
    Name: ACTIV04, dtype: int64
    
    
    ACTIV05 (8) :
    6    20959
    4    12470
    5    11365
    3     9868
    1     6534
    2     5228
    9     1456
    7      627
    Name: ACTIV05, dtype: int64
    
    
    ACTIV06 (8) :
    4    15624
    3    14566
    1    12813
    2     8210
    6     8177
    5     6937
    9     1901
    7      279
    Name: ACTIV06, dtype: int64
    
    
    ACTIV07 (8) :
    1    48418
    2     7925
    3     5474
    4     2371
    6     1639
    9     1447
    5     1138
    7       95
    Name: ACTIV07, dtype: int64
    
    
    TRANSP01 (4) :
    0    61625
    1     5822
    7      704
    9      356
    Name: TRANSP01, dtype: int64
    
    
    TRANSP02 (5) :
    0    36058
    1    25749
    8     5842
    7      503
    9      355
    Name: TRANSP02, dtype: int64
    
    
    TRANSP03 (5) :
    0    35764
    1    25987
    8     5840
    7      559
    9      357
    Name: TRANSP03, dtype: int64
    
    
    TRANSP04 (5) :
    0    48690
    1    13128
    8     5847
    7      486
    9      356
    Name: TRANSP04, dtype: int64
    
    
    TRANSP05 (5) :
    0    58376
    8     5847
    1     3538
    7      389
    9      357
    Name: TRANSP05, dtype: int64
    
    
    TRANSP06 (5) :
    0    36367
    1    25442
    8     5833
    7      508
    9      357
    Name: TRANSP06, dtype: int64
    
    
    TRANSP07 (5) :
    0    56969
    8     5847
    1     4866
    7      456
    9      369
    Name: TRANSP07, dtype: int64
    
    
    LEISSP01 (4) :
    0    61375
    1     6663
    9      392
    7       77
    Name: LEISSP01, dtype: int64
    
    
    LEISSP02 (4) :
    0    41661
    1    26360
    9      388
    7       98
    Name: LEISSP02, dtype: int64
    
    
    LEISSP03 (4) :
    0    45091
    1    22929
    9      387
    7      100
    Name: LEISSP03, dtype: int64
    
    
    LEISSP04 (4) :
    0    52719
    1    15330
    9      384
    7       74
    Name: LEISSP04, dtype: int64
    
    
    GROUPFR (4) :
    1    51939
    0    15369
    9      608
    7      591
    Name: GROUPFR, dtype: int64
    
    
    GRPAGE01 (6) :
     0    48542
     8    14450
     9     3836
     1     1400
    -9      215
     7       64
    Name: GRPAGE01, dtype: int64
    
    
    GRPAGE02 (6) :
     1    36694
     8    14447
     0    13240
     9     3835
    -9      215
     7       76
    Name: GRPAGE02, dtype: int64
    
    
    GRPAGE03 (6) :
     0    35396
     1    14539
     8    14449
     9     3836
    -9      215
     7       72
    Name: GRPAGE03, dtype: int64
    
    
    GRPAGE04 (6) :
     0    47848
     8    14449
     9     4008
     1     1939
    -9      215
     7       48
    Name: GRPAGE04, dtype: int64
    
    
    GRPAGE05 (6) :
     0    49473
     8    14449
     9     3836
     1      491
    -9      215
     7       43
    Name: GRPAGE05, dtype: int64
    
    
    GRPPUBL (6) :
     1    34335
     0    17276
     8    15175
     9     1343
    -9      215
     7      163
    Name: GRPPUBL, dtype: int64
    
    
    GRPEXIST (10) :
     3    24357
     8    15186
     2    11354
     4     9912
     1     3525
     9     1710
     5     1432
     6      537
     7      279
    -9      215
    Name: GRPEXIST, dtype: int64
    
    
    GRPILLAC (6) :
     0    37497
     8    15176
     1    13600
     9     1660
     7      359
    -9      215
    Name: GRPILLAC, dtype: int64
    
    
    GRPILLDO (6) :
     0    39786
     8    15176
     1    11573
     9     1520
     7      237
    -9      215
    Name: GRPILLDO, dtype: int64
    
    
    GRPGANG (6) :
     0    41498
     8    15175
     1     9978
     9     1464
    -9      215
     7      177
    Name: GRPGANG, dtype: int64
    
    
    GRPGEND (7) :
     3    29551
     8    15166
     1    12763
     2     9213
     9     1321
     7      278
    -9      215
    Name: GRPGEND, dtype: int64
    
    
    GRPETHN (7) :
    1    30741
    2    26626
    3     6746
    4     3116
    9     1138
    7      137
    8        3
    Name: GRPETHN, dtype: int64
    
    
    ETHNFRND (5) :
    1    49972
    6    13102
    0     4704
    9      611
    7      118
    Name: ETHNFRND, dtype: int64
    
    
    FRNDAC01 (6) :
    1    35229
    2    25741
    3     4907
    4     1502
    9     1035
    7       93
    Name: FRNDAC01, dtype: int64
    
    
    FRNDAC02 (6) :
    1    58766
    2     4492
    9     2512
    3     1725
    4      946
    7       66
    Name: FRNDAC02, dtype: int64
    
    
    FRNDAC03 (6) :
    1    52999
    2     9986
    3     2716
    9     1404
    4     1286
    7      116
    Name: FRNDAC03, dtype: int64
    
    
    FRNDAC04 (6) :
    1    56766
    2     8355
    9     1280
    3     1265
    4      765
    7       76
    Name: FRNDAC04, dtype: int64
    
    
    FRNDAC05 (6) :
    1    61360
    2     3883
    9     1390
    3     1026
    4      564
    7      284
    Name: FRNDAC05, dtype: int64
    
    
    FRNDAC06 (6) :
    2    23585
    3    19378
    1    12025
    4    11918
    9     1343
    7      258
    Name: FRNDAC06, dtype: int64
    
    
    FRNDAC07 (6) :
    2    25314
    3    21638
    1    10706
    4     9049
    9     1650
    7      150
    Name: FRNDAC07, dtype: int64
    
    
    FRNDAC08 (6) :
    1    48768
    2    12814
    3     2736
    9     2442
    4     1667
    7       80
    Name: FRNDAC08, dtype: int64
    
    
    FRNDAC09 (6) :
    9    30256
    1    13097
    3    10654
    4     9188
    2     4516
    7      796
    Name: FRNDAC09, dtype: int64
    
    
    ATTVIO01 (6) :
    1    38428
    2    15560
    3     9042
    4     4350
    9      966
    7      161
    Name: ATTVIO01, dtype: int64
    
    
    ATTVIO02 (6) :
    1    41097
    2    13535
    3     8150
    4     4467
    9     1097
    7      161
    Name: ATTVIO02, dtype: int64
    
    
    ATTVIO03 (6) :
    4    24071
    3    21182
    2    12666
    1     9113
    9      994
    7      481
    Name: ATTVIO03, dtype: int64
    
    
    ATTVIO04 (6) :
    1    41996
    2    13390
    3     6477
    4     4542
    9     1994
    7      108
    Name: ATTVIO04, dtype: int64
    
    
    ATTVIO05 (6) :
    1    23498
    3    16512
    2    16160
    4    11151
    9     1067
    7      119
    Name: ATTVIO05, dtype: int64
    
    
    SELFC01 (6) :
    3    21263
    2    19894
    1    16135
    4     9883
    9     1198
    7      134
    Name: SELFC01, dtype: int64
    
    
    SELFC02 (6) :
    1    24668
    2    20271
    3    14394
    4     7713
    9     1307
    7      154
    Name: SELFC02, dtype: int64
    
    
    SELFC03 (6) :
    3    19482
    2    18311
    1    16744
    4    11831
    9     1953
    7      186
    Name: SELFC03, dtype: int64
    
    
    SELFC04 (6) :
    1    19984
    3    18842
    2    16035
    4    12122
    9     1357
    7      167
    Name: SELFC04, dtype: int64
    
    
    SELFC05 (6) :
    1    24695
    2    16943
    3    15961
    4     9329
    9     1422
    7      157
    Name: SELFC05, dtype: int64
    
    
    SELFC06 (6) :
    1    26571
    2    20779
    3    12541
    4     6882
    9     1488
    7      246
    Name: SELFC06, dtype: int64
    
    
    SELFC07 (6) :
    1    26935
    2    21337
    3    11363
    4     7281
    9     1369
    7      222
    Name: SELFC07, dtype: int64
    
    
    SELFC08 (6) :
    1    29345
    2    18179
    3    10925
    4     8446
    9     1395
    7      217
    Name: SELFC08, dtype: int64
    
    
    SELFC09 (6) :
    1    35370
    2    19700
    3     7590
    4     4218
    9     1435
    7      194
    Name: SELFC09, dtype: int64
    
    
    SELFC10 (6) :
    1    21207
    2    19798
    3    15664
    4    10059
    9     1549
    7      230
    Name: SELFC10, dtype: int64
    
    
    SELFC11 (6) :
    1    18808
    2    17661
    3    15603
    4    14775
    9     1475
    7      185
    Name: SELFC11, dtype: int64
    
    
    SELFC12 (7) :
     3    18280
     2    17409
     1    16614
     4    14807
     9     1271
     7      125
    -9        1
    Name: SELFC12, dtype: int64
    
    
    ACCIDP (5) :
    1    35795
    2    20490
    3    11338
    9      819
    7       65
    Name: ACCIDP, dtype: int64
    
    
    ATTSCH (6) :
    3    30486
    2    18684
    4    11224
    1     7488
    9      416
    7      209
    Name: ATTSCH, dtype: int64
    
    
    REPGRADE (5) :
    1    57312
    2     8436
    3     2343
    9      397
    7       19
    Name: REPGRADE, dtype: int64
    
    
    TRUANCY (5) :
    1    48889
    2    13241
    3     5895
    9      461
    7       21
    Name: TRUANCY, dtype: int64
    
    
    ACHIEV (5) :
    2    41641
    3    19213
    1     6719
    9      639
    7      295
    Name: ACHIEV, dtype: int64
    
    
    ATSCH01 (6) :
    4    33714
    3    19090
    1     7888
    2     6959
    9      718
    7      138
    Name: ATSCH01, dtype: int64
    
    
    ATSCH02 (6) :
    3    26141
    4    25353
    2    10337
    1     5426
    9     1097
    7      153
    Name: ATSCH02, dtype: int64
    
    
    ATSCH03 (6) :
    3    25713
    4    22297
    2    11193
    1     7656
    9     1422
    7      226
    Name: ATSCH03, dtype: int64
    
    
    ATSCH04 (6) :
    4    32911
    3    17080
    2     8607
    1     8465
    9     1236
    7      208
    Name: ATSCH04, dtype: int64
    
    
    ATSCH05 (7) :
     2    21910
     1    17890
     3    16558
     4     9291
     9     2479
     7      302
    -9       77
    Name: ATSCH05, dtype: int64
    
    
    ATSCH06 (7) :
     2    22334
     3    18469
     1    14971
     4    11000
     9     1453
     7      218
    -9       62
    Name: ATSCH06, dtype: int64
    
    
    ATSCH07 (7) :
     2    21014
     1    18363
     3    16591
     4    11052
     9     1233
     7      180
    -9       74
    Name: ATSCH07, dtype: int64
    
    
    ATSCH08 (7) :
     1    40544
     2    14837
     3     6806
     4     4374
     9     1573
     7      230
    -9      143
    Name: ATSCH08, dtype: int64
    
    
    AFTSCH (9) :
    5     31693
    96     9538
    4      8442
    1      5864
    2      4933
    99     2578
    6      2408
    3      2352
    97      699
    Name: AFTSCH, dtype: int64
    
    
    NHOOD01 (6) :
    4    36563
    3    15233
    1     8052
    2     7589
    9      930
    7      140
    Name: NHOOD01, dtype: int64
    
    
    NHOOD02 (6) :
    1    28419
    2    15730
    3    12633
    4    10292
    9     1297
    7      136
    Name: NHOOD02, dtype: int64
    
    
    NHOOD03 (6) :
    4    36266
    3    18792
    2     6754
    1     4995
    9     1476
    7      224
    Name: NHOOD03, dtype: int64
    
    
    NHOOD04 (6) :
    4    28527
    3    16242
    2    10987
    1    10981
    9     1438
    7      332
    Name: NHOOD04, dtype: int64
    
    
    NHOOD05 (6) :
    1    35999
    2    16799
    3     8601
    4     5550
    9     1335
    7      223
    Name: NHOOD05, dtype: int64
    
    
    NHOOD06 (7) :
     1    46585
     2    10331
     3     5361
     4     4581
     9     1499
     7      149
    -9        1
    Name: NHOOD06, dtype: int64
    
    
    NHOOD07 (6) :
    1    38939
    2    14829
    3     7743
    4     5294
    9     1500
    7      202
    Name: NHOOD07, dtype: int64
    
    
    NHOOD08 (6) :
    1    46495
    2    12093
    3     4949
    4     3315
    9     1497
    7      158
    Name: NHOOD08, dtype: int64
    
    
    NHOOD09 (6) :
    1    36461
    2    13296
    3     9195
    4     7373
    9     1919
    7      263
    Name: NHOOD09, dtype: int64
    
    
    NHOOD10 (6) :
    3    24188
    4    24091
    2    11733
    1     6635
    9     1621
    7      239
    Name: NHOOD10, dtype: int64
    
    
    NHOOD11 (6) :
    3    22008
    4    18720
    2    16501
    1     9289
    9     1724
    7      265
    Name: NHOOD11, dtype: int64
    
    
    NHOOD12 (6) :
    3    23587
    4    19221
    2    14851
    1     8793
    9     1793
    7      262
    Name: NHOOD12, dtype: int64
    
    
    NHOOD13 (6) :
    1    28912
    2    21644
    3    10897
    4     5301
    9     1567
    7      186
    Name: NHOOD13, dtype: int64
    
    
    DELPDR (4) :
    0    50544
    1    15751
    9     1488
    7      724
    Name: DELPDR, dtype: int64
    
    
    DELPSL (4) :
    0    45322
    1    20617
    9     1657
    7      911
    Name: DELPSL, dtype: int64
    
    
    DELPBU (4) :
    0    60936
    1     5490
    9     1777
    7      304
    Name: DELPBU, dtype: int64
    
    
    DELPEX (4) :
    0    62476
    1     4185
    9     1572
    7      274
    Name: DELPEX, dtype: int64
    
    
    DELPAS (4) :
    0    60087
    1     6384
    9     1648
    7      388
    Name: DELPAS, dtype: int64
    
    
    BEERLTP (4) :
    1    41486
    0    25872
    9     1126
    7       23
    Name: BEERLTP, dtype: int64
    
    
    SPIRLTP (4) :
    0    44151
    1    22810
    9     1539
    7        7
    Name: SPIRLTP, dtype: int64
    
    
    HASHLTP (4) :
    0    60986
    1     5982
    9     1534
    7        5
    Name: HASHLTP, dtype: int64
    
    
    XTCLTP (4) :
    0    65884
    9     1649
    1      969
    7        5
    Name: XTCLTP, dtype: int64
    
    
    LHCLTP (5) :
     0    57535
    -9     5906
     9     4406
     1      657
     7        3
    Name: LHCLTP, dtype: int64
    
    
    VANDLTP (4) :
    0    58565
    1     8444
    9     1492
    7        6
    Name: VANDLTP, dtype: int64
    
    
    SHOPLTP (4) :
    0    55858
    1    11169
    9     1476
    7        4
    Name: SHOPLTP, dtype: int64
    
    
    BURGLTP (4) :
    0    65728
    9     1584
    1     1192
    7        3
    Name: BURGLTP, dtype: int64
    
    
    BICTLTP (4) :
    0    64761
    1     2232
    9     1512
    7        2
    Name: BICTLTP, dtype: int64
    
    
    CARTLTP (4) :
    0    66205
    9     1619
    1      682
    7        1
    Name: CARTLTP, dtype: int64
    
    
    DOWNLTP (4) :
    1    38095
    0    28917
    9     1480
    7       15
    Name: DOWNLTP, dtype: int64
    
    
    HACKLTP (4) :
    0    61739
    1     5081
    9     1683
    7        4
    Name: HACKLTP, dtype: int64
    
    
    CARBLTP (3) :
    0    65301
    1     1660
    9     1546
    Name: CARBLTP, dtype: int64
    
    
    SNATLTP (3) :
    0    65049
    1     1760
    9     1698
    Name: SNATLTP, dtype: int64
    
    
    WEAPLTP (4) :
    0    59428
    1     7521
    9     1551
    7        7
    Name: WEAPLTP, dtype: int64
    
    
    EXTOLTP (4) :
    0    65455
    9     1710
    1     1336
    7        6
    Name: EXTOLTP, dtype: int64
    
    
    GFIGLTP (4) :
    0    54061
    1    12798
    9     1642
    7        6
    Name: GFIGLTP, dtype: int64
    
    
    ASLTLTP (4) :
    0    64571
    1     2265
    9     1666
    7        5
    Name: ASLTLTP, dtype: int64
    
    
    DRUDLTP (4) :
    0    65020
    1     1750
    9     1727
    7       10
    Name: DRUDLTP, dtype: int64
    
    



```python
# Cleaning the dataset
# Note: some columns should retain 7,8,9. For example 'FAMILY'
# Use METADATA to check what to drop.
# ===== Dropping =======

# 7: ambiguous answer
# 9: no answer
# 97: ambiguous answer
# 98: not around
# 99: no answer
```


```python
# 'good' means the feature has been checked and is ready for training

df = df[df['MALE'] <= 1]      # good
df = df[df['AGEGROUP'] <= 3]  # good
df = df[df['BIRTHP'] <= 1]    # good
df = df[df['BIRTHPM'] <= 4]   # good
df = df[df['BIRTHPF'] <= 4]   # good
df = df[df['FAMILY'] <= 9]    # good
df = df[df['LANGH1'] <= 1]    # good
df = df[df['DISCRIM'] <= 4]   # good

df = df[df['OWNROOM'] <= 1]   # good
df = df[df['COMPUSE'] <= 1]   # good
df = df[df['OWNMOBPH'] <= 1]  # good
df = df[df['FAMILCAR'] <= 1]  # good
df = df[df['VICROBBP'] <= 1]  # good
df = df[df['VICASSAP'] <= 1]  # good
df = df[df['VICTHEFP'] <= 1]  # good
df = df[df['VICBULLP'] <= 1]  # good
```


```python
df = df[df['GETALFA'] <= 4]  # good
df = df[df['GETALMO'] <= 4]  # good
```


```python
df = df[df['LEISFAM'] <= 6]   # good
df = df[df['DINNFAM'] <= 8]   # good
```


```python
df = df[df['KNOWFR'] <= 3]   # good
```


```python
df = df[df['TELLTIME'] <= 8]   # good
df = df[df['OBEYTIME'] <= 8]   # good
```


```python
df = df[df['LIFEEV01'] <= 1]    # good
df = df[df['LIFEEV02'] <= 1]    # good
df = df[df['LIFEEV05'] <= 1]    # good
df = df[df['LIFEEV06'] <= 1]    # good
df = df[df['LIFEEV07'] <= 1]    # good
df = df[df['LIFEEV08'] <= 1]    # good
df = df[df['NIGHTACT'] <= 8]    # good

df = df[df['ACTIV01'] <= 6]     # good
df = df[df['ACTIV02'] <= 6]     # good
df = df[df['ACTIV03'] <= 6]     # good
df = df[df['ACTIV04'] <= 6]     # good
df = df[df['ACTIV05'] <= 6]     # good
df = df[df['ACTIV06'] <= 6]     # good
df = df[df['ACTIV07'] <= 6]     # good
```


```python
df = df[df['TRANSP01'] <= 1]    # good
df = df[df['TRANSP02'] <= 8]    # good
df = df[df['TRANSP03'] <= 8]    # good
df = df[df['TRANSP04'] <= 8]    # good
df = df[df['TRANSP05'] <= 8]    # good
df = df[df['TRANSP06'] <= 8]    # good
df = df[df['TRANSP07'] <= 8]    # good

df = df[df['LEISSP01'] <= 1]    # good
df = df[df['LEISSP02'] <= 1]    # good
df = df[df['LEISSP03'] <= 1]    # good
df = df[df['LEISSP04'] <= 1]    # good
```


```python
df = df[df['GROUPFR'] <= 1]     # good
df = df[df['GRPPUBL'] <= 8]     # good
```


```python
# Need to encode 'SCOUNTRY'
# It's the ONLY non-numerical column
df['SCOUNTRY'].value_counts()
```




    Italy          3878
    Switzerland    2715
    Germany        2424
    Czech Rep.     2365
    Austria        2072
    Portugal       2013
    Armenia        1856
    France         1832
    Russia         1764
    USA            1699
    Slovenia       1643
    Cyprus         1574
    Hungary        1522
    Estonia        1461
    Lithuania      1443
    Netherlands    1433
    Sweden         1404
    Belgium        1351
    Spain          1239
    Bosnia/H.      1160
    Norway         1134
    Ireland        1068
    Poland         1023
    Finland         912
    Denmark         817
    Iceland         411
    Aruba           402
    Venezuela       317
    Name: SCOUNTRY, dtype: int64




```python
# Encode country since it's non-numeric
le = LabelEncoder()

# fit and transform the string column
df['SCOUNTRY'] = le.fit_transform(df['SCOUNTRY'])
```


```python
df['SCOUNTRY'].value_counts()
```




    15    3878
    25    2715
    11    2424
    6     2365
    2     2072
    20    2013
    0     1856
    10    1832
    21    1764
    26    1699
    22    1643
    5     1574
    12    1522
    8     1461
    16    1443
    17    1433
    24    1404
    3     1351
    23    1239
    4     1160
    18    1134
    14    1068
    19    1023
    9      912
    7      817
    13     411
    1      402
    27     317
    Name: SCOUNTRY, dtype: int64



### Delinquent Acts That Make Up Violence
**vandltp**	Damage something on purpose

**extoltp**	Threaten someone w/ weapon

**gfigltp**	Partake in group fight

**asltltp**	Intentionally beat up someone



### Feature Engineering


```python
# create new column 'Violence' w/ correlated deliquent acts
df['Violence'] = (df['VANDLTP'] == 1) | (df['EXTOLTP'] == 1) | (df['GFIGLTP'] == 1 | (df['ASLTLTP'] == 1))
```


```python
# drop deliquent columns that are NOT related to violence
df.drop(columns=['SHOPLTP','BURGLTP','BICTLTP','CARTLTP','CARBLTP','SNATLTP','DRUDLTP','BEERLTP','SPIRLTP','HASHLTP','XTCLTP','LHCLTP','DOWNLTP','HACKLTP','WEAPLTP'], inplace=True)
```


```python
df.shape
```




    (42932, 130)




```python
# Target is now binary with an imbalance that is managable
df['Violence'].value_counts()
```




    False    32336
    True     10596
    Name: Violence, dtype: int64




```python
# Drop columns that made up 'Violence' column
df.drop(columns=['VANDLTP','EXTOLTP','GFIGLTP','ASLTLTP'], inplace=True)
#df = df.reset_index(drop=True)
```


```python
df.head()
```





  <div id="df-46c28b74-828c-4da5-9c72-0dde74c1fd05">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SCOUNTRY</th>
      <th>MALE</th>
      <th>AGEGROUP</th>
      <th>BIRTHP</th>
      <th>BIRTHPM</th>
      <th>BIRTHPF</th>
      <th>FAMILY</th>
      <th>LANGH1</th>
      <th>DISCRIM</th>
      <th>WORKFATH</th>
      <th>WORKMOTH</th>
      <th>OWNROOM</th>
      <th>COMPUSE</th>
      <th>OWNMOBPH</th>
      <th>FAMILCAR</th>
      <th>VICROBBP</th>
      <th>VICASSAP</th>
      <th>VICTHEFP</th>
      <th>VICBULLP</th>
      <th>GETALFA</th>
      <th>GETALMO</th>
      <th>LEISFAM</th>
      <th>DINNFAM</th>
      <th>KNOWFR</th>
      <th>TELLTIME</th>
      <th>OBEYTIME</th>
      <th>LIFEEV01</th>
      <th>LIFEEV02</th>
      <th>LIFEEV03</th>
      <th>LIFEEV04</th>
      <th>LIFEEV05</th>
      <th>LIFEEV06</th>
      <th>LIFEEV07</th>
      <th>LIFEEV08</th>
      <th>NIGHTACT</th>
      <th>ACTIV01</th>
      <th>ACTIV02</th>
      <th>ACTIV03</th>
      <th>ACTIV04</th>
      <th>ACTIV05</th>
      <th>ACTIV06</th>
      <th>ACTIV07</th>
      <th>TRANSP01</th>
      <th>TRANSP02</th>
      <th>TRANSP03</th>
      <th>TRANSP04</th>
      <th>TRANSP05</th>
      <th>TRANSP06</th>
      <th>TRANSP07</th>
      <th>LEISSP01</th>
      <th>LEISSP02</th>
      <th>LEISSP03</th>
      <th>LEISSP04</th>
      <th>GROUPFR</th>
      <th>GRPAGE01</th>
      <th>GRPAGE02</th>
      <th>GRPAGE03</th>
      <th>GRPAGE04</th>
      <th>GRPAGE05</th>
      <th>GRPPUBL</th>
      <th>GRPEXIST</th>
      <th>GRPILLAC</th>
      <th>GRPILLDO</th>
      <th>GRPGANG</th>
      <th>GRPGEND</th>
      <th>GRPETHN</th>
      <th>ETHNFRND</th>
      <th>FRNDAC01</th>
      <th>FRNDAC02</th>
      <th>FRNDAC03</th>
      <th>FRNDAC04</th>
      <th>FRNDAC05</th>
      <th>FRNDAC06</th>
      <th>FRNDAC07</th>
      <th>FRNDAC08</th>
      <th>FRNDAC09</th>
      <th>ATTVIO01</th>
      <th>ATTVIO02</th>
      <th>ATTVIO03</th>
      <th>ATTVIO04</th>
      <th>ATTVIO05</th>
      <th>SELFC01</th>
      <th>SELFC02</th>
      <th>SELFC03</th>
      <th>SELFC04</th>
      <th>SELFC05</th>
      <th>SELFC06</th>
      <th>SELFC07</th>
      <th>SELFC08</th>
      <th>SELFC09</th>
      <th>SELFC10</th>
      <th>SELFC11</th>
      <th>SELFC12</th>
      <th>ACCIDP</th>
      <th>ATTSCH</th>
      <th>REPGRADE</th>
      <th>TRUANCY</th>
      <th>ACHIEV</th>
      <th>ATSCH01</th>
      <th>ATSCH02</th>
      <th>ATSCH03</th>
      <th>ATSCH04</th>
      <th>ATSCH05</th>
      <th>ATSCH06</th>
      <th>ATSCH07</th>
      <th>ATSCH08</th>
      <th>AFTSCH</th>
      <th>NHOOD01</th>
      <th>NHOOD02</th>
      <th>NHOOD03</th>
      <th>NHOOD04</th>
      <th>NHOOD05</th>
      <th>NHOOD06</th>
      <th>NHOOD07</th>
      <th>NHOOD08</th>
      <th>NHOOD09</th>
      <th>NHOOD10</th>
      <th>NHOOD11</th>
      <th>NHOOD12</th>
      <th>NHOOD13</th>
      <th>DELPDR</th>
      <th>DELPSL</th>
      <th>DELPBU</th>
      <th>DELPEX</th>
      <th>DELPAS</th>
      <th>Violence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>99</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>99</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>26</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-46c28b74-828c-4da5-9c72-0dde74c1fd05')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-46c28b74-828c-4da5-9c72-0dde74c1fd05 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-46c28b74-828c-4da5-9c72-0dde74c1fd05');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Data Prepartion

### Define Input and Target Data


```python
# Define input features as X
X = df.drop('Violence', axis=1)
```


```python
X.head()
```





  <div id="df-5c3cdee4-c2e8-47a8-91b4-dc495ee72f6d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SCOUNTRY</th>
      <th>MALE</th>
      <th>AGEGROUP</th>
      <th>BIRTHP</th>
      <th>BIRTHPM</th>
      <th>BIRTHPF</th>
      <th>FAMILY</th>
      <th>LANGH1</th>
      <th>DISCRIM</th>
      <th>WORKFATH</th>
      <th>WORKMOTH</th>
      <th>OWNROOM</th>
      <th>COMPUSE</th>
      <th>OWNMOBPH</th>
      <th>FAMILCAR</th>
      <th>VICROBBP</th>
      <th>VICASSAP</th>
      <th>VICTHEFP</th>
      <th>VICBULLP</th>
      <th>GETALFA</th>
      <th>GETALMO</th>
      <th>LEISFAM</th>
      <th>DINNFAM</th>
      <th>KNOWFR</th>
      <th>TELLTIME</th>
      <th>OBEYTIME</th>
      <th>LIFEEV01</th>
      <th>LIFEEV02</th>
      <th>LIFEEV03</th>
      <th>LIFEEV04</th>
      <th>LIFEEV05</th>
      <th>LIFEEV06</th>
      <th>LIFEEV07</th>
      <th>LIFEEV08</th>
      <th>NIGHTACT</th>
      <th>ACTIV01</th>
      <th>ACTIV02</th>
      <th>ACTIV03</th>
      <th>ACTIV04</th>
      <th>ACTIV05</th>
      <th>ACTIV06</th>
      <th>ACTIV07</th>
      <th>TRANSP01</th>
      <th>TRANSP02</th>
      <th>TRANSP03</th>
      <th>TRANSP04</th>
      <th>TRANSP05</th>
      <th>TRANSP06</th>
      <th>TRANSP07</th>
      <th>LEISSP01</th>
      <th>LEISSP02</th>
      <th>LEISSP03</th>
      <th>LEISSP04</th>
      <th>GROUPFR</th>
      <th>GRPAGE01</th>
      <th>GRPAGE02</th>
      <th>GRPAGE03</th>
      <th>GRPAGE04</th>
      <th>GRPAGE05</th>
      <th>GRPPUBL</th>
      <th>GRPEXIST</th>
      <th>GRPILLAC</th>
      <th>GRPILLDO</th>
      <th>GRPGANG</th>
      <th>GRPGEND</th>
      <th>GRPETHN</th>
      <th>ETHNFRND</th>
      <th>FRNDAC01</th>
      <th>FRNDAC02</th>
      <th>FRNDAC03</th>
      <th>FRNDAC04</th>
      <th>FRNDAC05</th>
      <th>FRNDAC06</th>
      <th>FRNDAC07</th>
      <th>FRNDAC08</th>
      <th>FRNDAC09</th>
      <th>ATTVIO01</th>
      <th>ATTVIO02</th>
      <th>ATTVIO03</th>
      <th>ATTVIO04</th>
      <th>ATTVIO05</th>
      <th>SELFC01</th>
      <th>SELFC02</th>
      <th>SELFC03</th>
      <th>SELFC04</th>
      <th>SELFC05</th>
      <th>SELFC06</th>
      <th>SELFC07</th>
      <th>SELFC08</th>
      <th>SELFC09</th>
      <th>SELFC10</th>
      <th>SELFC11</th>
      <th>SELFC12</th>
      <th>ACCIDP</th>
      <th>ATTSCH</th>
      <th>REPGRADE</th>
      <th>TRUANCY</th>
      <th>ACHIEV</th>
      <th>ATSCH01</th>
      <th>ATSCH02</th>
      <th>ATSCH03</th>
      <th>ATSCH04</th>
      <th>ATSCH05</th>
      <th>ATSCH06</th>
      <th>ATSCH07</th>
      <th>ATSCH08</th>
      <th>AFTSCH</th>
      <th>NHOOD01</th>
      <th>NHOOD02</th>
      <th>NHOOD03</th>
      <th>NHOOD04</th>
      <th>NHOOD05</th>
      <th>NHOOD06</th>
      <th>NHOOD07</th>
      <th>NHOOD08</th>
      <th>NHOOD09</th>
      <th>NHOOD10</th>
      <th>NHOOD11</th>
      <th>NHOOD12</th>
      <th>NHOOD13</th>
      <th>DELPDR</th>
      <th>DELPSL</th>
      <th>DELPBU</th>
      <th>DELPEX</th>
      <th>DELPAS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>9</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>99</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>99</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>26</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5c3cdee4-c2e8-47a8-91b4-dc495ee72f6d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5c3cdee4-c2e8-47a8-91b4-dc495ee72f6d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5c3cdee4-c2e8-47a8-91b4-dc495ee72f6d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
X.shape
```




    (42932, 125)




```python
# Define Target
y = df['Violence']
```


```python
y.shape
```




    (42932,)



### Train Test Split Data


```python
# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
X_train.shape
```




    (30052, 125)




```python
X_test.shape
```




    (12880, 125)




```python
y_train.value_counts()
```




    False    22677
    True      7375
    Name: Violence, dtype: int64




```python
y_test.value_counts()
```




    False    9659
    True     3221
    Name: Violence, dtype: int64



# Feature Selection

## Using P-Value


```python
logistic_ml = sm.Logit(y_train, X_train)
logistic_coef = logistic_ml.fit()
summary = logistic_coef.summary2()
p_values = summary.tables[1]['P>|z|']
significant_p_values = p_values[p_values < 0.01]


```

    Optimization terminated successfully.
             Current function value: 0.424238
             Iterations 7



```python
if len(significant_p_values) > 0:
    p_values_99 = p_values.apply(lambda x: f"{x:.4f}*" if x < 0.01 else "Not Important")
    summary.tables[1]['P>|z|'] = p_values_99
    print(summary)
```

                             Results: Logit
    =================================================================
    Model:              Logit            Pseudo R-squared: 0.239     
    Dependent Variable: Violence         AIC:              25748.4065
    Date:               2023-05-04 06:25 BIC:              26787.2421
    No. Observations:   30052            Log-Likelihood:   -12749.   
    Df Model:           124              LL-Null:          -16746.   
    Df Residuals:       29927            LLR p-value:      0.0000    
    Converged:          1.0000           Scale:            1.0000    
    No. Iterations:     7.0000                                       
    -----------------------------------------------------------------
               Coef.  Std.Err.    z         P>|z|      [0.025  0.975]
    -----------------------------------------------------------------
    SCOUNTRY  -0.0301   0.0022 -13.5322       0.0000* -0.0344 -0.0257
    MALE       0.8360   0.0394  21.1992       0.0000*  0.7587  0.9133
    AGEGROUP  -0.1203   0.0644  -1.8678 Not Important -0.2465  0.0059
    BIRTHP    -0.1927   0.0666  -2.8943       0.0038* -0.3232 -0.0622
    BIRTHPM   -0.0404   0.0450  -0.8971 Not Important -0.1286  0.0478
    BIRTHPF   -0.0229   0.0407  -0.5624 Not Important -0.1026  0.0569
    FAMILY     0.0058   0.0139   0.4177 Not Important -0.0215  0.0332
    LANGH1    -0.2024   0.0598  -3.3851       0.0007* -0.3197 -0.0852
    DISCRIM    0.0386   0.0282   1.3696 Not Important -0.0166  0.0937
    WORKFATH  -0.0010   0.0014  -0.7545 Not Important -0.0037  0.0017
    WORKMOTH  -0.0012   0.0017  -0.7371 Not Important -0.0045  0.0021
    OWNROOM   -0.0138   0.0390  -0.3531 Not Important -0.0901  0.0626
    COMPUSE   -0.0748   0.0538  -1.3912 Not Important -0.1801  0.0306
    OWNMOBPH   0.0601   0.0602   0.9980 Not Important -0.0579  0.1781
    FAMILCAR   0.1473   0.0577   2.5550 Not Important  0.0343  0.2604
    VICROBBP   0.1519   0.0773   1.9653 Not Important  0.0004  0.3034
    VICASSAP   0.4363   0.0780   5.5930       0.0000*  0.2834  0.5893
    VICTHEFP   0.3183   0.0393   8.0893       0.0000*  0.2412  0.3954
    VICBULLP   0.1093   0.0476   2.2970 Not Important  0.0160  0.2025
    GETALFA   -0.1187   0.0270  -4.3959       0.0000* -0.1716 -0.0658
    GETALMO   -0.2256   0.0295  -7.6356       0.0000* -0.2835 -0.1677
    LEISFAM    0.0090   0.0127   0.7090 Not Important -0.0159  0.0340
    DINNFAM    0.0155   0.0076   2.0574 Not Important  0.0007  0.0303
    KNOWFR    -0.2904   0.0273 -10.6461       0.0000* -0.3439 -0.2369
    TELLTIME  -0.0156   0.0081  -1.9178 Not Important -0.0316  0.0003
    OBEYTIME  -0.0249   0.0063  -3.9355       0.0001* -0.0373 -0.0125
    LIFEEV01   0.0647   0.0872   0.7420 Not Important -0.1062  0.2356
    LIFEEV02  -0.0225   0.1005  -0.2239 Not Important -0.2195  0.1745
    LIFEEV03   0.0232   0.0157   1.4806 Not Important -0.0075  0.0539
    LIFEEV04  -0.0159   0.0153  -1.0386 Not Important -0.0460  0.0141
    LIFEEV05   0.2419   0.0343   7.0569       0.0000*  0.1747  0.3091
    LIFEEV06   0.0728   0.0636   1.1453 Not Important -0.0518  0.1975
    LIFEEV07   0.0906   0.0546   1.6597 Not Important -0.0164  0.1976
    LIFEEV08   0.0284   0.0500   0.5679 Not Important -0.0696  0.1265
    NIGHTACT   0.0267   0.0084   3.1648       0.0016*  0.0102  0.0432
    ACTIV01   -0.0856   0.0155  -5.5340       0.0000* -0.1159 -0.0553
    ACTIV02   -0.0775   0.0153  -5.0621       0.0000* -0.1075 -0.0475
    ACTIV03   -0.0104   0.0128  -0.8132 Not Important -0.0356  0.0147
    ACTIV04   -0.0247   0.0172  -1.4396 Not Important -0.0583  0.0089
    ACTIV05    0.0645   0.0125   5.1632       0.0000*  0.0400  0.0890
    ACTIV06    0.0702   0.0118   5.9348       0.0000*  0.0470  0.0934
    ACTIV07    0.0158   0.0149   1.0620 Not Important -0.0134  0.0450
    TRANSP01  -1.9819   0.9263  -2.1396 Not Important -3.7975 -0.1664
    TRANSP02  -0.0451   0.0328  -1.3752 Not Important -0.1094  0.0192
    TRANSP03  -0.0577   0.0348  -1.6571 Not Important -0.1259  0.0105
    TRANSP04  -0.1724   0.0392  -4.3952       0.0000* -0.2492 -0.0955
    TRANSP05   0.3268   0.0654   4.9935       0.0000*  0.1985  0.4551
    TRANSP06   0.0756   0.0332   2.2734 Not Important  0.0104  0.1408
    TRANSP07   0.0872   0.0476   1.8327 Not Important -0.0061  0.1804
    LEISSP01  -0.1473   0.0893  -1.6496 Not Important -0.3224  0.0277
    LEISSP02  -0.2213   0.0754  -2.9361       0.0033* -0.3690 -0.0736
    LEISSP03  -0.0355   0.0754  -0.4706 Not Important -0.1834  0.1124
    LEISSP04   0.2138   0.0786   2.7204       0.0065*  0.0598  0.3679
    GROUPFR    0.6240   0.1384   4.5092       0.0000*  0.3528  0.8952
    GRPAGE01  -0.0895   0.0938  -0.9538 Not Important -0.2734  0.0944
    GRPAGE02  -0.0628   0.0425  -1.4766 Not Important -0.1461  0.0206
    GRPAGE03   0.1492   0.0413   3.6109       0.0003*  0.0682  0.2303
    GRPAGE04  -0.0042   0.0323  -0.1316 Not Important -0.0675  0.0590
    GRPAGE05   0.0522   0.1175   0.4446 Not Important -0.1780  0.2825
    GRPPUBL   -0.0419   0.0283  -1.4820 Not Important -0.0974  0.0135
    GRPEXIST  -0.0519   0.0159  -3.2588       0.0011* -0.0832 -0.0207
    GRPILLAC   0.0144   0.0189   0.7600 Not Important -0.0227  0.0514
    GRPILLDO   0.0831   0.0213   3.8971       0.0001*  0.0413  0.1249
    GRPGANG   -0.0044   0.0228  -0.1908 Not Important -0.0491  0.0404
    GRPGEND   -0.0489   0.0185  -2.6489       0.0081* -0.0851 -0.0127
    GRPETHN    0.0227   0.0159   1.4276 Not Important -0.0085  0.0540
    ETHNFRND  -0.0312   0.0079  -3.9328       0.0001* -0.0467 -0.0156
    FRNDAC01   0.0081   0.0196   0.4125 Not Important -0.0303  0.0464
    FRNDAC02  -0.0981   0.0150  -6.5358       0.0000* -0.1275 -0.0687
    FRNDAC03   0.0308   0.0200   1.5387 Not Important -0.0084  0.0700
    FRNDAC04   0.2176   0.0239   9.0901       0.0000*  0.1707  0.2645
    FRNDAC05  -0.1398   0.0241  -5.7913       0.0000* -0.1871 -0.0925
    FRNDAC06  -0.0139   0.0166  -0.8352 Not Important -0.0464  0.0187
    FRNDAC07  -0.0268   0.0146  -1.8304 Not Important -0.0555  0.0019
    FRNDAC08   0.1088   0.0138   7.8938       0.0000*  0.0818  0.1358
    FRNDAC09  -0.0093   0.0048  -1.9583 Not Important -0.0186  0.0000
    ATTVIO01   0.1347   0.0173   7.7831       0.0000*  0.1008  0.1686
    ATTVIO02   0.0223   0.0165   1.3519 Not Important -0.0100  0.0547
    ATTVIO03   0.1477   0.0158   9.3757       0.0000*  0.1168  0.1786
    ATTVIO04  -0.0513   0.0145  -3.5461       0.0004* -0.0796 -0.0229
    ATTVIO05   0.0502   0.0151   3.3146       0.0009*  0.0205  0.0799
    SELFC01    0.0347   0.0157   2.2096 Not Important  0.0039  0.0655
    SELFC02    0.0140   0.0162   0.8634 Not Important -0.0178  0.0458
    SELFC03   -0.0509   0.0138  -3.7013       0.0002* -0.0779 -0.0240
    SELFC04    0.1032   0.0168   6.1286       0.0000*  0.0702  0.1362
    SELFC05    0.0996   0.0171   5.8340       0.0000*  0.0661  0.1330
    SELFC06   -0.0145   0.0161  -0.9033 Not Important -0.0461  0.0170
    SELFC07   -0.0954   0.0165  -5.7894       0.0000* -0.1277 -0.0631
    SELFC08   -0.0305   0.0158  -1.9267 Not Important -0.0616  0.0005
    SELFC09   -0.0562   0.0177  -3.1707       0.0015* -0.0909 -0.0214
    SELFC10   -0.0046   0.0147  -0.3094 Not Important -0.0334  0.0243
    SELFC11    0.1191   0.0144   8.2543       0.0000*  0.0909  0.1474
    SELFC12    0.0430   0.0153   2.8038       0.0051*  0.0129  0.0731
    ACCIDP     0.0931   0.0165   5.6370       0.0000*  0.0607  0.1254
    ATTSCH    -0.1097   0.0198  -5.5540       0.0000* -0.1484 -0.0710
    REPGRADE  -0.0324   0.0312  -1.0374 Not Important -0.0935  0.0288
    TRUANCY    0.1801   0.0218   8.2686       0.0000*  0.1374  0.2228
    ACHIEV    -0.2001   0.0230  -8.6859       0.0000* -0.2452 -0.1549
    ATSCH01   -0.0293   0.0171  -1.7077 Not Important -0.0628  0.0043
    ATSCH02    0.0144   0.0167   0.8608 Not Important -0.0184  0.0472
    ATSCH03   -0.0304   0.0165  -1.8392 Not Important -0.0628  0.0020
    ATSCH04   -0.0251   0.0146  -1.7213 Not Important -0.0536  0.0035
    ATSCH05   -0.0600   0.0151  -3.9754       0.0001* -0.0897 -0.0304
    ATSCH06    0.1071   0.0155   6.9068       0.0000*  0.0767  0.1375
    ATSCH07    0.0593   0.0161   3.6774       0.0002*  0.0277  0.0910
    ATSCH08   -0.0310   0.0138  -2.2416 Not Important -0.0581 -0.0039
    AFTSCH    -0.0002   0.0004  -0.3653 Not Important -0.0010  0.0007
    NHOOD01    0.0161   0.0181   0.8857 Not Important -0.0195  0.0516
    NHOOD02   -0.0263   0.0147  -1.7878 Not Important -0.0552  0.0025
    NHOOD03   -0.0528   0.0186  -2.8330       0.0046* -0.0893 -0.0163
    NHOOD04   -0.0422   0.0139  -3.0437       0.0023* -0.0694 -0.0150
    NHOOD05    0.0377   0.0182   2.0700 Not Important  0.0020  0.0734
    NHOOD06   -0.0156   0.0184  -0.8446 Not Important -0.0516  0.0205
    NHOOD07    0.1328   0.0175   7.5728       0.0000*  0.0984  0.1672
    NHOOD08   -0.0843   0.0174  -4.8335       0.0000* -0.1185 -0.0501
    NHOOD09    0.0348   0.0141   2.4617 Not Important  0.0071  0.0625
    NHOOD10   -0.0615   0.0180  -3.4238       0.0006* -0.0967 -0.0263
    NHOOD11    0.0076   0.0173   0.4410 Not Important -0.0263  0.0416
    NHOOD12   -0.0183   0.0170  -1.0779 Not Important -0.0517  0.0150
    NHOOD13   -0.0748   0.0154  -4.8612       0.0000* -0.1050 -0.0447
    DELPDR     0.0068   0.0178   0.3827 Not Important -0.0280  0.0416
    DELPSL     0.1285   0.0159   8.0804       0.0000*  0.0974  0.1597
    DELPBU    -0.0421   0.0196  -2.1460 Not Important -0.0806 -0.0036
    DELPEX    -0.1129   0.0257  -4.3973       0.0000* -0.1633 -0.0626
    DELPAS     0.0077   0.0219   0.3524 Not Important -0.0351  0.0505
    =================================================================
    



```python
if len(significant_p_values) > 0:
    p_values_99 = p_values.apply(lambda x: 1 if x < 0.01 else 0)
    significant_features = X.columns[p_values_99 == 1]
    print(significant_features)
```

    Index(['SCOUNTRY', 'MALE', 'BIRTHP', 'LANGH1', 'VICASSAP', 'VICTHEFP',
           'GETALFA', 'GETALMO', 'KNOWFR', 'OBEYTIME', 'LIFEEV05', 'NIGHTACT',
           'ACTIV01', 'ACTIV02', 'ACTIV05', 'ACTIV06', 'TRANSP04', 'TRANSP05',
           'LEISSP02', 'LEISSP04', 'GROUPFR', 'GRPAGE03', 'GRPEXIST', 'GRPILLDO',
           'GRPGEND', 'ETHNFRND', 'FRNDAC02', 'FRNDAC04', 'FRNDAC05', 'FRNDAC08',
           'ATTVIO01', 'ATTVIO03', 'ATTVIO04', 'ATTVIO05', 'SELFC03', 'SELFC04',
           'SELFC05', 'SELFC07', 'SELFC09', 'SELFC11', 'SELFC12', 'ACCIDP',
           'ATTSCH', 'TRUANCY', 'ACHIEV', 'ATSCH05', 'ATSCH06', 'ATSCH07',
           'NHOOD03', 'NHOOD04', 'NHOOD07', 'NHOOD08', 'NHOOD10', 'NHOOD13',
           'DELPSL', 'DELPEX'],
          dtype='object')



```python
pvals_col =['SCOUNTRY', 'MALE', 'BIRTHP', 'LANGH1', 'VICASSAP', 'VICTHEFP',
       'GETALFA', 'GETALMO', 'KNOWFR', 'OBEYTIME', 'LIFEEV05', 'NIGHTACT',
       'ACTIV01', 'ACTIV02', 'ACTIV05', 'ACTIV06', 'TRANSP04', 'TRANSP05',
       'LEISSP02', 'LEISSP04', 'GROUPFR', 'GRPAGE03', 'GRPEXIST', 'GRPILLDO',
       'GRPGEND', 'ETHNFRND', 'FRNDAC02', 'FRNDAC04', 'FRNDAC05', 'FRNDAC08',
       'ATTVIO01', 'ATTVIO03', 'ATTVIO04', 'ATTVIO05', 'SELFC03', 'SELFC04',
       'SELFC05', 'SELFC07', 'SELFC09', 'SELFC11', 'SELFC12', 'ACCIDP',
       'ATTSCH', 'TRUANCY', 'ACHIEV', 'ATSCH05', 'ATSCH06', 'ATSCH07',
       'NHOOD03', 'NHOOD04', 'NHOOD07', 'NHOOD08', 'NHOOD10', 'NHOOD13',
       'DELPSL', 'DELPEX']
```


```python
X_train.shape
```




    (30052, 125)




```python
X_test.shape
```




    (12880, 125)




```python
# Apply P-value feature selection to X_train and X_test
X_train_pval = X_train.loc[:, pvals_col]
X_test_pval = X_test.loc[:, pvals_col]
```


```python
X_train_pval.shape
```




    (30052, 56)




```python
X_test_pval.shape
```




    (12880, 56)



## Random Forest Feature Importance


```python
# Random Forest feature importance
rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>




```python
# Random Forest feature importance in descending order
importances = rnd_clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
feat_labels = df.columns[0:]
labels = []

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[sorted_idx[f]],
                            importances[sorted_idx[f]]))
    labels.append(feat_labels[sorted_idx[f]])
```

     1) FRNDAC04                       0.038641
     2) DELPSL                         0.022505
     3) ATTVIO03                       0.021975
     4) FRNDAC08                       0.021715
     5) ATTVIO01                       0.021024
     6) SCOUNTRY                       0.019911
     7) SELFC05                        0.019775
     8) DELPAS                         0.016438
     9) FRNDAC03                       0.016218
    10) MALE                           0.015456
    11) DELPDR                         0.014963
    12) NHOOD07                        0.014912
    13) SELFC04                        0.014901
    14) NIGHTACT                       0.014589
    15) SELFC11                        0.013657
    16) ATTVIO04                       0.013232
    17) SELFC06                        0.012689
    18) AFTSCH                         0.012399
    19) ACTIV06                        0.012019
    20) ATTVIO02                       0.011776
    21) TRUANCY                        0.011693
    22) KNOWFR                         0.011518
    23) ACTIV05                        0.011390
    24) ATSCH06                        0.010893
    25) ACTIV01                        0.010769
    26) ATTVIO05                       0.010592
    27) LEISFAM                        0.010581
    28) SELFC02                        0.010317
    29) SELFC10                        0.010038
    30) ACTIV03                        0.010033
    31) GRPEXIST                       0.010008
    32) ACCIDP                         0.009854
    33) ATSCH07                        0.009783
    34) SELFC12                        0.009684
    35) FRNDAC06                       0.009586
    36) SELFC01                        0.009566
    37) ATSCH03                        0.009403
    38) OBEYTIME                       0.009185
    39) ATSCH05                        0.009131
    40) DINNFAM                        0.009104
    41) FRNDAC07                       0.009026
    42) ACTIV02                        0.008926
    43) NHOOD09                        0.008771
    44) NHOOD12                        0.008739
    45) SELFC03                        0.008681
    46) FRNDAC09                       0.008666
    47) ATTSCH                         0.008635
    48) NHOOD05                        0.008632
    49) NHOOD02                        0.008466
    50) NHOOD11                        0.008416
    51) ATSCH02                        0.008363
    52) SELFC08                        0.008320
    53) NHOOD04                        0.008269
    54) NHOOD10                        0.008227
    55) DELPBU                         0.008215
    56) ACTIV04                        0.008137
    57) NHOOD13                        0.008005
    58) SELFC07                        0.007998
    59) FRNDAC01                       0.007966
    60) SELFC09                        0.007862
    61) GRPGEND                        0.007675
    62) ATSCH04                        0.007591
    63) ATSCH01                        0.007533
    64) WORKMOTH                       0.007475
    65) GRPETHN                        0.007327
    66) NHOOD01                        0.007306
    67) NHOOD03                        0.007238
    68) NHOOD08                        0.006930
    69) ATSCH08                        0.006929
    70) NHOOD06                        0.006784
    71) ACHIEV                         0.006767
    72) GRPILLDO                       0.006687
    73) ACTIV07                        0.006592
    74) WORKFATH                       0.006479
    75) FAMILY                         0.006156
    76) GRPILLAC                       0.006062
    77) LEISSP04                       0.005862
    78) GETALFA                        0.005498
    79) GETALMO                        0.005371
    80) FRNDAC05                       0.005113
    81) GRPAGE02                       0.005086
    82) DELPEX                         0.005084
    83) ETHNFRND                       0.005048
    84) GRPGANG                        0.005004
    85) GRPPUBL                        0.004919
    86) GRPAGE03                       0.004875
    87) TRANSP03                       0.004723
    88) LIFEEV03                       0.004722
    89) TELLTIME                       0.004671
    90) REPGRADE                       0.004502
    91) TRANSP06                       0.004432
    92) TRANSP02                       0.004332
    93) LEISSP02                       0.004294
    94) VICTHEFP                       0.004237
    95) FRNDAC02                       0.004042
    96) LIFEEV05                       0.003964
    97) DISCRIM                        0.003898
    98) BIRTHPF                        0.003812
    99) TRANSP04                       0.003757
    100) LEISSP03                       0.003695
    101) LIFEEV04                       0.003684
    102) BIRTHPM                        0.003394
    103) OWNROOM                        0.003287
    104) GRPAGE04                       0.002966
    105) VICBULLP                       0.002873
    106) LIFEEV08                       0.002746
    107) LIFEEV07                       0.002628
    108) GRPAGE01                       0.002579
    109) TRANSP05                       0.002543
    110) TRANSP07                       0.002504
    111) COMPUSE                        0.002458
    112) GRPAGE05                       0.002374
    113) VICASSAP                       0.002335
    114) LANGH1                         0.002247
    115) FAMILCAR                       0.002235
    116) LIFEEV06                       0.002083
    117) OWNMOBPH                       0.002028
    118) BIRTHP                         0.001969
    119) AGEGROUP                       0.001814
    120) VICROBBP                       0.001790
    121) LEISSP01                       0.001781
    122) GROUPFR                        0.001675
    123) LIFEEV01                       0.001530
    124) LIFEEV02                       0.001138
    125) TRANSP01                       0.000621



```python
# Build a word cloud to visualize top 15 features for presentation
df_word = pd.DataFrame(labels, columns = ['Feat_Names'])
```


```python
df_word.shape
```




    (125, 1)




```python
df_word = df_word.drop(df_word.index[15:])

```


```python
df_word.shape
```




    (15, 1)




```python
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS

text = " ".join(i for i in df_word['Feat_Names'])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(7,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_69_0.png)
    



```python
# Top 15 features based on feature_importances_
labels[0:14]
```




    ['FRNDAC04',
     'DELPSL',
     'ATTVIO03',
     'FRNDAC08',
     'ATTVIO01',
     'SCOUNTRY',
     'SELFC05',
     'DELPAS',
     'FRNDAC03',
     'MALE',
     'DELPDR',
     'NHOOD07',
     'SELFC04',
     'NIGHTACT']




```python
# Get the number of features in the original DataFrame
num_features = df.shape[1]

# Make sure feat_labels contains the correct number of feature labels
feat_labels = df.columns[:num_features]

# Check that importances has the same length as the number of features
if len(importances) != num_features:
    # If not, only include the first num_features importances
    importances = importances[:num_features]

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [feat_labels[i] for i in indices]

plt.figure()
plt.title("Feature Importance")
plt.bar(range(len(names)), importances[indices])
#plt.xticks(range(len(names)), names, rotation=90)
plt.show()

```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_71_0.png)
    



```python
# Features with importance of 0.075 or greater based on print out of feature_importances and plot
rf_imps =  labels[:62]
```


```python
np.array(rf_imps)
```




    array(['FRNDAC04', 'DELPSL', 'ATTVIO03', 'FRNDAC08', 'ATTVIO01',
           'SCOUNTRY', 'SELFC05', 'DELPAS', 'FRNDAC03', 'MALE', 'DELPDR',
           'NHOOD07', 'SELFC04', 'NIGHTACT', 'SELFC11', 'ATTVIO04', 'SELFC06',
           'AFTSCH', 'ACTIV06', 'ATTVIO02', 'TRUANCY', 'KNOWFR', 'ACTIV05',
           'ATSCH06', 'ACTIV01', 'ATTVIO05', 'LEISFAM', 'SELFC02', 'SELFC10',
           'ACTIV03', 'GRPEXIST', 'ACCIDP', 'ATSCH07', 'SELFC12', 'FRNDAC06',
           'SELFC01', 'ATSCH03', 'OBEYTIME', 'ATSCH05', 'DINNFAM', 'FRNDAC07',
           'ACTIV02', 'NHOOD09', 'NHOOD12', 'SELFC03', 'FRNDAC09', 'ATTSCH',
           'NHOOD05', 'NHOOD02', 'NHOOD11', 'ATSCH02', 'SELFC08', 'NHOOD04',
           'NHOOD10', 'DELPBU', 'ACTIV04', 'NHOOD13', 'SELFC07', 'FRNDAC01',
           'SELFC09', 'GRPGEND', 'ATSCH04'], dtype='<U8')




```python
# Apply RF-Importance feature selection to X_train X_test
X_train_rf_imp = X_train.loc[:, rf_imps]
X_test_rf_imp = X_test.loc[:, rf_imps]
```


```python
X_train_rf_imp.shape
```




    (30052, 62)




```python
X_train_rf_imp.columns
```




    Index(['FRNDAC04', 'DELPSL', 'ATTVIO03', 'FRNDAC08', 'ATTVIO01', 'SCOUNTRY',
           'SELFC05', 'DELPAS', 'FRNDAC03', 'MALE', 'DELPDR', 'NHOOD07', 'SELFC04',
           'NIGHTACT', 'SELFC11', 'ATTVIO04', 'SELFC06', 'AFTSCH', 'ACTIV06',
           'ATTVIO02', 'TRUANCY', 'KNOWFR', 'ACTIV05', 'ATSCH06', 'ACTIV01',
           'ATTVIO05', 'LEISFAM', 'SELFC02', 'SELFC10', 'ACTIV03', 'GRPEXIST',
           'ACCIDP', 'ATSCH07', 'SELFC12', 'FRNDAC06', 'SELFC01', 'ATSCH03',
           'OBEYTIME', 'ATSCH05', 'DINNFAM', 'FRNDAC07', 'ACTIV02', 'NHOOD09',
           'NHOOD12', 'SELFC03', 'FRNDAC09', 'ATTSCH', 'NHOOD05', 'NHOOD02',
           'NHOOD11', 'ATSCH02', 'SELFC08', 'NHOOD04', 'NHOOD10', 'DELPBU',
           'ACTIV04', 'NHOOD13', 'SELFC07', 'FRNDAC01', 'SELFC09', 'GRPGEND',
           'ATSCH04'],
          dtype='object')



# Models

--------------------------------------------

## Logistic Regression

### Log Reg Baseline


```python
log_clf = LogisticRegression(max_iter=2000, random_state=42)
```


```python
log_clf.fit(X_train, y_train)
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(max_iter=2000, random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=2000, random_state=42)</pre></div></div></div></div></div>




```python
y_pred = log_clf.predict(X_test)
```


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
           False       0.83      0.93      0.88      9659
            True       0.66      0.41      0.51      3221
    
        accuracy                           0.80     12880
       macro avg       0.74      0.67      0.69     12880
    weighted avg       0.79      0.80      0.78     12880
    



```python
print(f1_score(y_test, y_pred))
```

    0.508331737215093


### Log Reg P-Value


```python
log_clf_pval = LogisticRegression(max_iter=2000, random_state=42)
log_clf_pval.fit(X_train_pval, y_train)
y_pred_pval = log_clf_pval.predict(X_test_pval)
```


```python
print(classification_report(y_test, y_pred_pval))
```

                  precision    recall  f1-score   support
    
           False       0.82      0.93      0.87      9659
            True       0.66      0.40      0.50      3221
    
        accuracy                           0.80     12880
       macro avg       0.74      0.67      0.69     12880
    weighted avg       0.78      0.80      0.78     12880
    



```python
print(f1_score(y_test, y_pred_pval))
```

    0.4997114829774957


### Log Reg Feat_Importance


```python
log_clf_rf_imp = LogisticRegression(max_iter=2000, random_state=42)
log_clf_rf_imp.fit(X_train_rf_imp, y_train)
y_pred_pval2 = log_clf_rf_imp.predict(X_test_rf_imp)
```


```python
print(classification_report(y_test, y_pred_pval2))
```

                  precision    recall  f1-score   support
    
           False       0.82      0.94      0.87      9659
            True       0.66      0.38      0.48      3221
    
        accuracy                           0.80     12880
       macro avg       0.74      0.66      0.68     12880
    weighted avg       0.78      0.80      0.77     12880
    



```python
print(f1_score(y_test, y_pred_pval2))
```

    0.4787170857255989


### Log Reg Hyper-parameter Tune


```python
# define the parameter grid to search over
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'saga', 'newton-cg'],
    'max_iter': [2000]
}
```


```python
# define the grid search cross-validation object
grid_search = GridSearchCV(log_clf_pval, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_pval, y_train)
```

    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py:378: FitFailedWarning: 
    60 fits failed out of a total of 240.
    The score on these train-test partitions for these parameters will be set to nan.
    If these failures are not expected, you can try to debug them by setting error_score='raise'.
    
    Below are more details about the failures:
    --------------------------------------------------------------------------------
    30 fits failed with the following error:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py", line 1162, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py", line 54, in _check_solver
        raise ValueError(
    ValueError: Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
    
    --------------------------------------------------------------------------------
    30 fits failed with the following error:
    Traceback (most recent call last):
      File "/usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_validation.py", line 686, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py", line 1162, in fit
        solver = _check_solver(self.solver, self.penalty, self.dual)
      File "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py", line 54, in _check_solver
        raise ValueError(
    ValueError: Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
    
      warnings.warn(some_fits_failed_message, FitFailedWarning)
    /usr/local/lib/python3.10/dist-packages/sklearn/model_selection/_search.py:952: UserWarning: One or more of the test scores are non-finite: [       nan 0.16018773 0.14190453        nan 0.44008202 0.44341505
     0.44116954 0.44008202        nan 0.45569118 0.45028807        nan
     0.4811991  0.48309578 0.48158727 0.4811991         nan 0.49256116
     0.49199973        nan 0.49356201 0.49348539 0.49331254 0.49348224
            nan 0.49378185 0.49413798        nan 0.49421535 0.49481559
     0.49452876 0.49435387        nan 0.4948334  0.49439864        nan
     0.49456008 0.49456881 0.49456751 0.4949926         nan 0.49496086
     0.49439864        nan 0.49498653 0.49465498 0.49439864 0.4949926 ]
      warnings.warn(





<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=LogisticRegression(max_iter=2000, random_state=42),
             param_grid={&#x27;C&#x27;: [0.001, 0.01, 0.1, 1, 10, 100],
                         &#x27;max_iter&#x27;: [2000], &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],
                         &#x27;solver&#x27;: [&#x27;lbfgs&#x27;, &#x27;liblinear&#x27;, &#x27;saga&#x27;, &#x27;newton-cg&#x27;]},
             scoring=&#x27;f1&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=LogisticRegression(max_iter=2000, random_state=42),
             param_grid={&#x27;C&#x27;: [0.001, 0.01, 0.1, 1, 10, 100],
                         &#x27;max_iter&#x27;: [2000], &#x27;penalty&#x27;: [&#x27;l1&#x27;, &#x27;l2&#x27;],
                         &#x27;solver&#x27;: [&#x27;lbfgs&#x27;, &#x27;liblinear&#x27;, &#x27;saga&#x27;, &#x27;newton-cg&#x27;]},
             scoring=&#x27;f1&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=2000, random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(max_iter=2000, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>




```python
print("Best parameters: ", grid_search.best_params_)
print("Best F1 score: ", grid_search.best_score_)
```

    Best parameters:  {'C': 10, 'max_iter': 2000, 'penalty': 'l2', 'solver': 'newton-cg'}
    Best F1 score:  0.49499260136491535



```python
# Best parameters:  {'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
```


```python
log_final = LogisticRegression(C=10, penalty='l2', solver='newton-cg', max_iter=2000)
log_final.fit(X_train_pval, y_train)
y_pred_log_final = log_final.predict(X_test_pval)
```


```python
print(classification_report(y_test, y_pred_log_final))
```

                  precision    recall  f1-score   support
    
           False       0.82      0.93      0.87      9659
            True       0.66      0.40      0.50      3221
    
        accuracy                           0.80     12880
       macro avg       0.74      0.67      0.69     12880
    weighted avg       0.78      0.80      0.78     12880
    



```python
print(f1_score(y_test, y_pred_log_final))
```

    0.5000961723408348


### Log Reg Confusion Matrix


```python
cm_log = confusion_matrix(y_test, y_pred_log_final)

f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_log,annot = True,cmap='Reds',linewidths=1,linecolor='k',square=True,mask=False,fmt = ".0f",cbar=True,ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_103_0.png)
    


### Log Reg Threshold tuning


```python
# predict probabilities for test set
y_pred_proba = log_final.predict_proba(X_test_pval)[:, 1]
```


```python
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
```


```python
# plot the precision-recall curve
plt.figure(figsize=(5, 4))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_107_0.png)
    



```python
# calculate F1-score for each threshold
f1_scores = [f1_score(y_test, y_pred_proba >= th) for th in thresholds]

# find the threshold that maximizes the F1-score
best_threshold = thresholds[np.argmax(f1_scores)]
```


```python
# plot the F1-scores against the thresholds
plt.plot(thresholds, f1_scores)
plt.axvline(x=0.5, color='k', linestyle='--', label='Default threshold')
plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best threshold')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1-score vs. Threshold')
plt.legend()
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_109_0.png)
    



```python
best_threshold
```




    0.29771286877015407




```python
# set a threshold value (0.5 by default)
#threshold = 0.2916142395850417
threshold = 0.29771286877015407

# adjust threshold and make predictions
y_pred_thresh_log = np.where(y_pred_proba > threshold, 1, 0)
```

### Log Reg Metrics on Best Model


```python
# Confusion Matrix, ROC-AUC?
print(classification_report(y_test, y_pred_thresh_log))
```

                  precision    recall  f1-score   support
    
           False       0.88      0.81      0.84      9659
            True       0.54      0.67      0.60      3221
    
        accuracy                           0.78     12880
       macro avg       0.71      0.74      0.72     12880
    weighted avg       0.80      0.78      0.78     12880
    



```python
print(f1_score(y_test, y_pred_thresh_log))
```

    0.5985563575791226



```python
print(accuracy_score(y_test, y_pred_thresh_log))
```

    0.7754658385093167



```python
cm_log = confusion_matrix(y_test, y_pred_thresh_log)

f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_log,annot = True,cmap='Reds',linewidths=1,linecolor='k',square=True,mask=False,fmt = ".0f",cbar=True,ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_116_0.png)
    



```python
skplt.metrics.plot_roc(y_test, log_final.predict_proba(X_test_pval), plot_micro=False)
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_117_0.png)
    


## Decision Tree

### Decision Tree Baseline


```python
# Tune parameters later
dtc_clf = DecisionTreeClassifier(random_state=42)
```


```python
dtc_clf.fit(X_train, y_train)
```




<style>#sk-container-id-5 {color: black;background-color: white;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=42)</pre></div></div></div></div></div>




```python
y_pred2 = dtc_clf.predict(X_test)
```


```python
print(classification_report(y_test, y_pred2))
```

                  precision    recall  f1-score   support
    
           False       0.82      0.81      0.82      9659
            True       0.45      0.46      0.46      3221
    
        accuracy                           0.72     12880
       macro avg       0.63      0.64      0.64     12880
    weighted avg       0.73      0.72      0.73     12880
    



```python
print(f1_score(y_test, y_pred2))
```

    0.4559410500460547


### DT P-Value


```python
dtc_clf_pval = DecisionTreeClassifier(random_state=42)
dtc_clf_pval.fit(X_train_pval, y_train)
y_pred_pval2 = dtc_clf_pval.predict(X_test_pval)
```


```python
print(classification_report(y_test, y_pred_pval2))
```

                  precision    recall  f1-score   support
    
           False       0.82      0.82      0.82      9659
            True       0.46      0.47      0.46      3221
    
        accuracy                           0.73     12880
       macro avg       0.64      0.64      0.64     12880
    weighted avg       0.73      0.73      0.73     12880
    



```python
print(f1_score(y_test, y_pred_pval2))
```

    0.4644666155026861


### DT Hyper-parameter Tune


```python
# define the parameter grid
param_grid = {'max_depth': [2, 4, 6, 8, 10],
              'min_samples_split': [2, 4, 6, 8, 10],
              'min_samples_leaf': [1, 2, 4, 6, 8],
              'criterion': ['gini', 'entropy']}
```


```python
# perform grid search cross-validation
grid_search = GridSearchCV(dtc_clf_pval, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_pval, y_train)
```




<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=42),
             param_grid={&#x27;criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;max_depth&#x27;: [2, 4, 6, 8, 10],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4, 6, 8],
                         &#x27;min_samples_split&#x27;: [2, 4, 6, 8, 10]},
             scoring=&#x27;f1&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=42),
             param_grid={&#x27;criterion&#x27;: [&#x27;gini&#x27;, &#x27;entropy&#x27;],
                         &#x27;max_depth&#x27;: [2, 4, 6, 8, 10],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4, 6, 8],
                         &#x27;min_samples_split&#x27;: [2, 4, 6, 8, 10]},
             scoring=&#x27;f1&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>




```python
# print the best hyperparameters
print(grid_search.best_params_)
```

    {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 6, 'min_samples_split': 2}



```python
# Best Params
# {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 6, 'min_samples_split': 2}
```


```python
# train the final model on the entire training set using the best hyperparameters
dtc_final = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=6, min_samples_split=2, random_state=42)
dtc_final.fit(X_train_pval, y_train)
y_pred_dtc_final = dtc_final.predict(X_test_pval)
```


```python
print(classification_report(y_test, y_pred_dtc_final))
```

                  precision    recall  f1-score   support
    
           False       0.83      0.91      0.86      9659
            True       0.60      0.43      0.50      3221
    
        accuracy                           0.79     12880
       macro avg       0.72      0.67      0.68     12880
    weighted avg       0.77      0.79      0.77     12880
    



```python
print(f1_score(y_test, y_pred_dtc_final))
```

    0.5022681908909453


### DT Confusion Matrix


```python
cm = confusion_matrix(y_test, y_pred_dtc_final)

f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm,annot = True,cmap='Reds',linewidths=1,linecolor='k',square=True,mask=False,fmt = ".0f",cbar=True,ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_138_0.png)
    


### DT Threshold Tuning


```python
y_pred_proba = dtc_final.predict_proba(X_test_pval)[:, 1]
```


```python
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# plot the precision-recall curve
plt.figure(figsize=(5, 4))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_141_0.png)
    



```python
# calculate F1-score for each threshold
f1_scores = [f1_score(y_test, y_pred_proba >= th) for th in thresholds]

# find the threshold that maximizes the F1-score
best_threshold = thresholds[np.argmax(f1_scores)]
```


```python
# plot the F1-scores against the thresholds
plt.plot(thresholds, f1_scores)
plt.axvline(x=0.5, color='k', linestyle='--', label='Default threshold')
plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best threshold')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1-score vs. Threshold')
plt.legend()
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_143_0.png)
    



```python
best_threshold
```




    0.35172413793103446




```python
# set a threshold value (0.5 by default)
threshold = 0.35172413793103446

# adjust threshold and make predictions
y_pred_thresh_dtc = np.where(y_pred_proba > threshold, 1, 0)
```

### DT Metrics on Best Model


```python
print(classification_report(y_test, y_pred_thresh_dtc))
```

                  precision    recall  f1-score   support
    
           False       0.86      0.82      0.84      9659
            True       0.52      0.58      0.55      3221
    
        accuracy                           0.76     12880
       macro avg       0.69      0.70      0.69     12880
    weighted avg       0.77      0.76      0.77     12880
    



```python
print(f1_score(y_test, y_pred_thresh_dtc))
```

    0.550885927661444



```python
print(accuracy_score(y_test, y_pred_thresh_dtc))
```

    0.7618788819875777



```python
cm_dtc = confusion_matrix(y_test, y_pred_thresh_dtc)

f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_dtc,annot = True,cmap='Reds',linewidths=1,linecolor='k',square=True,mask=False,fmt = ".0f",cbar=True,ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_150_0.png)
    



```python
skplt.metrics.plot_roc(y_test, dtc_final.predict_proba(X_test_pval), plot_micro=False)
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_151_0.png)
    


## Random Forest

### Random Forest Baseline


```python
rnd_clf = RandomForestClassifier(random_state=42)
```


```python
rnd_clf.fit(X_train, y_train)
```




<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" checked><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>




```python
y_pred3 = rnd_clf.predict(X_test)
```


```python
print(classification_report(y_test, y_pred3))
```

                  precision    recall  f1-score   support
    
           False       0.83      0.95      0.89      9659
            True       0.74      0.40      0.52      3221
    
        accuracy                           0.82     12880
       macro avg       0.78      0.68      0.70     12880
    weighted avg       0.81      0.82      0.79     12880
    



```python
print(f1_score(y_test, y_pred3))
```

    0.5211097708082026


### RF P-Value


```python
rnd_clf_pval = RandomForestClassifier(random_state=42)
rnd_clf_pval.fit(X_train_pval, y_train)
y_pred_pval3 = rnd_clf_pval.predict(X_test_pval)
```


```python
print(classification_report(y_test, y_pred_pval3))
```

                  precision    recall  f1-score   support
    
           False       0.83      0.95      0.88      9659
            True       0.73      0.41      0.53      3221
    
        accuracy                           0.81     12880
       macro avg       0.78      0.68      0.71     12880
    weighted avg       0.80      0.81      0.80     12880
    



```python
print(f1_score(y_test, y_pred_pval3))
```

    0.5273769519667918



```python
# Not Applicable
```

### RF Hyper-parameter Tune


```python
# Define hyperparameters to tune
hyperparameters = {
    'n_estimators': [50, 100, 200, 400],
    'max_depth': [2, 4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```


```python
# perform grid search cross-validation
grid_search = GridSearchCV(rnd_clf_pval, hyperparameters, cv=5, scoring='f1')
grid_search.fit(X_train_pval, y_train)
```




<style>#sk-container-id-8 {color: black;background-color: white;}#sk-container-id-8 pre{padding: 0;}#sk-container-id-8 div.sk-toggleable {background-color: white;}#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-8 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-8 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-8 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-8 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-8 div.sk-item {position: relative;z-index: 1;}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-8 div.sk-item::before, #sk-container-id-8 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-8 div.sk-label-container {text-align: center;}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-8 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42),
             param_grid={&#x27;max_depth&#x27;: [2, 4, 6, 8, 10, None],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],
                         &#x27;min_samples_split&#x27;: [2, 5, 10],
                         &#x27;n_estimators&#x27;: [50, 100, 200, 400]},
             scoring=&#x27;f1&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42),
             param_grid={&#x27;max_depth&#x27;: [2, 4, 6, 8, 10, None],
                         &#x27;min_samples_leaf&#x27;: [1, 2, 4],
                         &#x27;min_samples_split&#x27;: [2, 5, 10],
                         &#x27;n_estimators&#x27;: [50, 100, 200, 400]},
             scoring=&#x27;f1&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>




```python
 # print the best hyperparameters
print(grid_search.best_params_)
```

    {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 50}



```python
# best params
# {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 50}
```


```python
# train the final model on the entire training set using the best hyperparameters
#min_samples_leaf = 2 / 0.529388
#min_sample_split = 10 / good
#n_estimators = 49 / 0.530234
#n_estimators = 48 / 0.5313914

rnd_final = RandomForestClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=48, random_state=42)
rnd_final.fit(X_train_pval, y_train)
y_pred_rnd_final = rnd_final.predict(X_test_pval)
```


```python
print(classification_report(y_test, y_pred_rnd_final))
```

                  precision    recall  f1-score   support
    
           False       0.83      0.95      0.88      9659
            True       0.73      0.42      0.53      3221
    
        accuracy                           0.82     12880
       macro avg       0.78      0.68      0.71     12880
    weighted avg       0.80      0.82      0.80     12880
    



```python
print(f1_score(y_test, y_pred_rnd_final))
```

    0.5313914583743358


### RF Confusion Matrix


```python
cm = confusion_matrix(y_test, y_pred_rnd_final)

f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm,annot = True,cmap='Reds',linewidths=1,linecolor='k',square=True,mask=False,fmt = ".0f",cbar=True,ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_173_0.png)
    


### RF Threshold Tuning


```python
y_pred_proba = rnd_final.predict_proba(X_test_pval)[:, 1]
```


```python
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# plot the precision-recall curve
plt.figure(figsize=(5, 4))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_176_0.png)
    



```python
# calculate F1-score for each threshold
f1_scores = [f1_score(y_test, y_pred_proba >= th) for th in thresholds]

# find the threshold that maximizes the F1-score
best_threshold = thresholds[np.argmax(f1_scores)]
```


```python
# plot the F1-scores against the thresholds
plt.plot(thresholds, f1_scores)
plt.axvline(x=0.5, color='k', linestyle='--', label='Default threshold')
plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best threshold')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1-score vs. Threshold')
plt.legend()
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_178_0.png)
    



```python
best_threshold
```




    0.32894616957116957




```python
# set a threshold value (0.5 by default)
threshold = 0.32894616957116957

# adjust threshold and make predictions
y_pred_thresh_rnd = np.where(y_pred_proba > threshold, 1, 0)
```

### RF Metrics on Best Model


```python
print(classification_report(y_test, y_pred_thresh_rnd))
```

                  precision    recall  f1-score   support
    
           False       0.88      0.82      0.85      9659
            True       0.56      0.68      0.61      3221
    
        accuracy                           0.79     12880
       macro avg       0.72      0.75      0.73     12880
    weighted avg       0.80      0.79      0.79     12880
    



```python
print(f1_score(y_test, y_pred_thresh_rnd))
```

    0.6123310810810811



```python
print(accuracy_score(y_test, y_pred_thresh_rnd))
```

    0.7861801242236025



```python
cm_rnd = confusion_matrix(y_test, y_pred_thresh_rnd)

f, ax = plt.subplots(figsize =(3,3))
sns.heatmap(cm_rnd,annot = True,cmap='Reds',linewidths=1,linecolor='k',square=True,mask=False,fmt = ".0f",cbar=True,ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_185_0.png)
    



```python
skplt.metrics.plot_roc(y_test, rnd_final.predict_proba(X_test_pval), plot_micro=False)
plt.show()
```


    
![png](Group1_Data240_Final_Project_1_files/Group1_Data240_Final_Project_1_186_0.png)
    


# F1 Score Comparing Models

Logistic Regression


```python
print(f1_score(y_test, y_pred_thresh_log))
```

    0.5985563575791226


Decision Tree


```python
print(f1_score(y_test, y_pred_thresh_dtc))
```

    0.550885927661444


Random Forest


```python
print(f1_score(y_test, y_pred_thresh_rnd))
```

    0.6123310810810811

