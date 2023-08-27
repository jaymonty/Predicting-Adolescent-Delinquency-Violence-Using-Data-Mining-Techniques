# Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques

19

**Predicting Adolescent Delinquency Violence Using Data Mining Techniques** Dhruv Jain, Edward Montoya, Himanshee, Jeet Pankajkumar Parekh

Department of Applied Data Science, San Jose State University DATA 240: Data Mining

Professor Seungjoon Lee

2023, May 14

**Contents**

1. [**Introduction 3**](#_page2_x72.00_y72.00)
1. [Motivation 3](#_page2_x72.00_y99.60)
1. [Literature Review 3](#_page2_x72.00_y430.77)
2. [**Dataset 7**](#_page6_x72.00_y72.00)
   1. [Dataset Description 7](#_page6_x72.00_y99.60)
   1. [Dataset Pre-Processing 8](#_page7_x72.00_y390.64)
3. [**Feature Understanding 10**](#_page9_x72.00_y154.79)
1. [Feature Selection 10](#_page9_x72.00_y265.18)
1. [Important Features (Extracted Knowledge) 13](#_page12_x72.00_y237.59)
4. [**Methodology 15**](#_page14_x72.00_y347.98)
   1. [Balance Data 15](#_page14_x72.00_y485.96)
   1. [Threshold Tuning 16](#_page15_x72.00_y265.18)
   1. [Primary Metric for Evaluation 16](#_page15_x72.00_y596.36)
   1. [Model Development 17](#_page16_x72.00_y375.57)
5. [**Results 18**](#_page17_x72.00_y324.49)
1. [Logistic Regression 18](#_page17_x72.00_y545.27)
1. [Decision Tree 21](#_page20_x72.00_y343.99)
1. [Random Forest 24](#_page23_x72.00_y310.39)
1. [Model Comparison 27](#_page26_x72.00_y591.08)
6. [**Discussion 28**](#_page27_x72.00_y452.43)
1. [Why One Method is Better? 28](#_page27_x72.00_y562.82)
1. [Why Selected Features Important? 29](#_page28_x72.00_y347.98)
1. [Meaning of Result 29](#_page28_x72.00_y623.95)

[**References 31**](#_page30_x72.00_y529.60)

1. **Introduction**
1. <a name="_page2_x72.00_y72.00"></a>**Motivation**

<a name="_page2_x72.00_y99.60"></a>We chose this topic because understanding child delinquency is an ongoing research issue. Based on various literature reviews on the subject, self-control for an adolescent can be predicted from various factors. Unfortunately, most of these papers focus on making predictions for immigrants, gender-focused, or a specific delinquent act. This project intends to broaden the scope and use data mining techniques to build a binary classification for predicting violence.

The target feature consists of multiple delinquent behaviors grouped into a single target called violence. Hence, this problem requires classification methods to predict violent behavior. Based on prior research, the primary means of classifying delinquent behavior has utilized Logistic Regression, PCA, and other forms of multivariate analysis. This project is a binary classification task that focuses on classifying if an adolescent will either commit violence or not by using Logistic Regression, Decision Tree, and Random Forest.

2. **Literature<a name="_page2_x72.00_y430.77"></a> Review**

In the paper by Sabia et al. (2017), the group presents the problem of a research gap in trying to predict delinquency for immigrant and non-immigrant adolescents. The group attributes this problem to the lack of studies that have been conducted on this particular topic. The paper aimed to expand upon previous studies on acculturation and delinquency. Specifically, the paper examined how social, individual, and environmental risk factors affect delinquency among adolescents from three different generational statuses. The research was guided by acculturation and social bond theories. The paper explored the impact of factors such as family bonding, school climate, self-control, neighborhood disorganization, and delinquent peer associations on first-generation, second-generation, and native-born delinquency. In the paper, the total number

of U.S. adolescents included in the sample was 2,091, comprising 83 first-generation immigrants, 287 second-generation immigrants, and 1,722 native-born youth. In the paper, the group used multiple regression analyses, demonstrating that various variables predicted delinquency for each generational status group. The results suggested that distinct risk factors such as self-control level for first-generation immigrants, self-control, neighborhood disorganization, and delinquent peers for second-generation immigrants, and a blend of psychosocial, individual, and environmental factors for native-born youth could be significant predictors of delinquency. The paper revealed that compared to first-generation immigrants, native-born and second-generation immigrant youth are more vulnerable to delinquency risk factors. The findings support previous research on acculturation theory and the immigrant paradox, which suggests that the resilience of first-generation immigrants to delinquency decreases across successive generations.

Enzmann et al. (2010) discuss the findings of the Second International Self-Report Delinquency Study (ISRD-2), which looked at delinquency and victimization among 12-15 year old students from 63 cities and 31 countries. The study examined the prevalence of different types of delinquency and victimization and found significant differences between country clusters, with Western European and Anglo-Saxon countries scoring the highest. However, the results for victimization did not follow this pattern. The article also compared ISRD-2 results with other international crime-related statistics and found moderate levels of convergence. The paper suggests that country clusters based on theoretical and policy-related criteria may be a promising data analysis approach.

In the paper by Posick (2013), they discuss the connection between those who commit crimes (offenders) and those who are victimized. Research has found that these two groups are not randomly distributed and tend to share similar characteristics and experiences with violence and deviance. The results showed a positive association between offending and victimization for violence and theft. The relationship remained significant even when other factors were considered, indicating that theories of crime can explain victimization as well. Self-control and deviant peers were the most powerful predictors of offending and victimization, but adverse life events and family bonding were more strongly associated with victimization.

Pauwels et al. (2011) investigate the relationship between societal vulnerability, violent values, low self-control, and involvement in troublesome youth groups among Flemish adolescents aged 14-18. The study found that societal vulnerability is positively associated with violent values and low self-control. In turn, violent values and low self-control are positively associated with involvement in troublesome youth groups. The study also found that violent values and low self-control mediate the relationship between societal vulnerability and troublesome youth group involvement. The findings suggest that interventions to reduce youth involvement in troublesome groups should address societal vulnerability, promote values that oppose violence, and increase self-control among young people. This could be achieved through social policies and targeted programs. The study contributes to the literature on youth delinquency and has important implications for policy and practice.

Stansfield (2015) investigates the link between teen participation in sports and risky behavior, emphasizing gender and cross-national variations. The study compares the risky behavior of adolescent sports players with non-participants across 36 nations using data from the Health Behaviour in School-aged Children survey. According to the study, teens who play sports are generally less likely to engage in dangerous activities like smoking, drinking, and using drugs. However, sports involvement and dangerous behavior are correlated differently by gender

and nation. The protective impact of participating in sports, in particular, differs among nations and is more pronounced for girls than for boys. The study provides important insights into the relationship between sports participation and adolescent health and highlights the need for interventions that consider gender and cultural differences in promoting adolescent health.

In the paper by Sabia (2017), she investigated a void in the literature concerning how risk factors predict self-control in both immigrant and nonimmigrant teenagers. Three research hypotheses were formulated to evaluate the relevance of self-control theory among different generational groups and the impact of family and environmental factors on adolescents' self-control development. Using data from the Second International Self-Reported Delinquency Study Dataset, a school-based sample of 2,056 American adolescents was analyzed through hierarchical multiple regression analyses. The sample comprised 4% first-generation immigrants, 13% second-generation immigrants, and 83% native-born individuals and was used to test the hypotheses. The findings partially supported the self-control theory, and they indicated that family bonding, parental supervision, neighborhood disorganization, school disorganization, and delinquent peer associations were similarly predictive of self-control across all three generational status groups. However, school climate was a significant predictor only for native-born adolescents. Additionally, the results demonstrated that various factors within the family, neighborhood, school, and peer environments contributed significantly to developing self-control in both immigrant and native-born youths. The results indicated that family and environmental factors play a role in shaping the development of self-control in youths. Nonetheless, the differing impact of these factors on self-control development among immigrant and nonimmigrant adolescents may be attributed to cultural variations fostered by acculturation.

2. **Dataset**
1. <a name="_page6_x72.00_y72.00"></a>**Dataset<a name="_page6_x72.00_y99.60"></a> Description**

The dataset is from the Second International Self-Reported Delinquency Study, conducted from 2005 to 2007. The dataset consists of 68507 rows and 708 columns. Each row represents a person filling out the questionnaire. The dataset was reduced to 144 columns based on domain knowledge. It then consists of 143 numerical columns and one categorical column. Figure 1 shows the sample of the dataset.

**Figure 1**

*Dataset Sample*


![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 001](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/912d9ceb-c704-4c65-a96b-e159747fa076)


The sample of the questionnaire can be found in Figure 2. This questionnaire was used to obtain the data.

**Figure 2**

*Sample of the Questionnaire*


![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 002](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/9ebeea05-f62f-445a-be83-08e6adb9c2d5)



2. **Dataset<a name="_page7_x72.00_y390.64"></a> Pre-Processing**

After reducing the dataset to 144 columns based on domain knowledge, two specific columns, 'Unnamed: 0' and 'CASEID,' were intentionally removed from our dataset. The 'Unnamed: 0' column was identified and dropped. This column acted primarily as an index, a role already fulfilled within our dataset. The existence of 'Unnamed: 0' was thus considered redundant, and its removal was instrumental in maintaining the simplicity and readability of the dataset. The 'CASEID' column, incorporated initially for identification during data collection, was also eliminated. This was due to its inability to contribute valuable information to the subsequent analysis and modeling tasks.

We filtered out rows based on certain conditions for each column. Each line with df = df[df['COLUMN\_NAME'] <= VALUE] is filtering the data frame to keep only the rows where the value in 'COLUMN\_NAME' is less than or equal to 'VALUE.' This is a way of removing no answers or ambiguous answers. For instance, df = df[df['MALE'] <= 1] ensures that we keep only the rows where the 'MALE' column's value is less than or equal to 1 to avoid no answers.

In our case, each feature column was carefully examined and modified as necessary to ensure its readiness for the subsequent stages of our project. The 'good' notation against each feature indicated that it was reviewed and deemed appropriate for the next step. These preprocessing measures often improve the accuracy and efficiency of the predictive models or analytics algorithms used later in the project.

The single object type required encoding preprocessing in our dataset. The 'SCOUNTRY' column in the dataset represents the country and is categorical (non-numerical). Before any analysis or modeling, converting these categorical data into numerical format is essential. We have applied label encoding using the LabelEncoder class from the sklearn to achieve this—preprocessing module.

The result is a transformed 'SCOUNTRY' column where a unique integer represents each country. This encoding allows the data to be used more effectively by machine learning algorithms.

In feature engineering, we created a new feature or column called 'Violence' in the data frame df. This new feature is based on four existing features: 'VANDLTP,' 'EXTOLTP,' 'GFIGLTP,' and 'ASLTLTP.' Each of these acts represents a different type of violence:

- 'VANDLTP': Damage something on purpose
- 'EXTOLTP': Threaten someone with a weapon
- 'GFIGLTP': Partake in a group fight
- 'ASLTLTP': Intentionally beating up someone

This newly created 'Violence' feature is then used as the primary target feature for further analysis or predictive modeling. This suggests that the analysis or model aims to understand or predict violence based on the other features in the dataset.

3. **Feature<a name="_page9_x72.00_y154.79"></a> Understanding**

The feature understanding section aims to identify a primary method to implement feature selection and to find a technique to extract hidden knowledge from the features backed by prior research.

1. **Feature<a name="_page9_x72.00_y265.18"></a> Selection**

This project explored two feature selection methods: Logistic Regression P-value and Random Forest feature importance (*feature\_importances\_*). The P-value method works by fitting a Logistic Regression model on the training data, which then estimates the coefficients of the logistic regression model using the maximum likelihood method (Hosmer & Lemeshow, 2000). Finally, the model is able to extract the P-values from the coefficients. For this process, we applied a 99% confidence rule on selecting features, which resulted in bringing the features from 125 to just 56 while maintaining a similar performance on the full dataset. A snapshot of this process can be seen in the figure below. Note that we labeled features below the 99% confidence level as “Not Important.”

**Figure 3**

*Snapshot of Results from Logistic Regression P-Value Feature Selection Method*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 003](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/eef93845-6885-4f6b-ba9f-e02da18bb40c)


When comparing the performance of this method to the full dataset, the F1 score went up slightly for the Decision Tree (0.0085 improvement) and Random Forest (0.0062 improvement). The Logistic Regression model performed slightly worse than the full dataset base model (0.0086 worse). Overall, this method reduced the dimensionality and maintained good performance.

The Random Forest feature importance works by representing the reduction in impurity achieved by using that feature for splitting the data at each tree node (Louppe, 2014). If the feature is important, it will have a higher impurity reduction. The feature importance scores are then normalized so that they sum to one, with higher scores indicating that they are more important features. Unfortunately, this method is more arbitrary in how to implement it for feature selection. We decided to visually plot all of the features with their corresponding feature importance value, which can be seen in the figure below.

**Figure 4**

*Visual Bar Chart of All Features and Their Feature Importance Score*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 004](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/c779b0bb-598c-4cb4-9c99-670bb7c6612d)


Based on our research, the threshold value of (0.01) is often used to select features with this method. We were concerned that this would result in poor performance due to the lack of features (about 40 features). Instead, we chose to utilize the value of (0.075) because we could observe a noticeable dip in the bar chart plot just after this value. This value equated to 62 features that were (0.075) or above in importance. We then used these 62 features to test our base models for all three models. Unfortunately, the results were poor for this method of feature selection. For example, when we compared this method with the P-value on the Logistic Regression model, we found that our performance on the F1 score metric had degraded by (0.021). This was also true for the Decision Tree and Random Forest.

We were surprised that the Logistic Regression P-value method performed best in            selecting the fewest features while maintaining performance. This is considering that the P-value method had six fewer features than the Random Forest feature importance method. For this reason, we decided to implement all of our models with the P-value method going forward and to use the feature importance method instead to derive hidden knowledge.

It is important to note that all of the top 15 important features were also selected by the Logistic Regression P-value feature selection method (56 features) for the purpose of building models.

2. **Important<a name="_page12_x72.00_y237.59"></a> Features (Extracted Knowledge)**

The Random Forest feature importance (*feature\_importances\_*) allowed the ability to quantify which features in our dataset were important with the intent to extract knowledge from the data. The figure below shows the top 15 features in our dataset of 125 features based on their feature importance score.

**Figure 5**

*Snapshot of the Top 15 Important Features Based on Random Forest feature\_importances\_*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 005](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/f364687d-c5a2-472e-b985-383b8a8ce719)


At first glance, it is challenging to derive any meaning from this snapshot due to the fact that the features are not very intuitive to understand. Upon closer examination, we begin to see that there exist two primary themes. The first theme is features related to friends. The following features are related to this theme:

- **FRNDAC04** (*Friends smash or vandalize things just for fun*)
- **DELPSL** (*I have friends who entered a building with the purpose to steal something*)
- **FRNDAC08** (*Friends frighten and annoy people around us just for fun*)
- **DELPAS** (*I have friends who did beat someone up or hurt someone badly with something like a stick or a knife*)
- **FRNDAC03** (*Friends drink a lot of beer/alcohol*)
- **DELPDR** (*I have friends who use soft or hard drugs like weed, hash, XTC, speed, heroin or coke*)
- **NIGHTACT** (*How many times a week do you usually go out at night, such as going to a party or a disco, go to somebody’s house or hanging out on the street*)

From these features, we can see that there’s a relationship between adolescents and committing violence because they are hanging around friends that partake in various delinquent acts. Based on domain knowledge, this relationship makes sense because things like peer pressure could potentially contribute to adolescents committing delinquent behavior.

The second theme that can be observed from feature importance is related to self-control. The following features are related to this theme:

- **ATTVIO03** (*If somebody attacks me, I will hit him/her back*)
- **ATTVIO01** (*A bit of violence is part of the fun*)
- **SELFC05** (*Sometimes I will take a risk just for the fun of it*)
- **SELFC04** (*I like to test myself every now and then by doing something a little risky*)
- **SELFC11** (*When I’m really angry, other people better stay away from me*)

Interestingly, both themes observed correlate to prior research. In the paper *Societal Vulnerability and Troublesome Youth Group Involvement* by Pauwels et al. (2011), the authors explore the relationship between societal vulnerability, violent values, low self-control, and involvement in troublesome youth groups among adolescents. The authors found that adolescents with violent values and a lack of self-control were more likely to have societal vulnerability. Meaning that these values and lack of control positively correlated to involvement in risky behavior, gangs, and committing violence. The authors concluded by discussing the impact that friends and self-control play in the role of contributing to such delinquent behavior. Finally, the authors recommend remedies that target interventions in reducing adolescents' involvement with gangs and violent values.

4. **Methodology**

<a name="_page14_x72.00_y347.98"></a>In the feature understanding section, we discussed how we would implement our models with the Logistic Regression P-value method for feature selection because it reduced the dimensionality and maintained great performance. The rest of our methodology will be discussed in this section.

1. **Balance<a name="_page14_x72.00_y485.96"></a> Data**

At the beginning of constructing models, we implemented two approaches for handling the imbalance of the target data. First, we applied the Synthetic Minority Oversampling TEchnique (SMOTE), which works by synthetically creating more examples of the minority target violence (Chawla et al., 2002). We then evaluated the performance of SMOTE on the base models. Then, we implemented undersampling of the majority target (non-violence) and evaluated the performance on the base models. Both models were successful in balancing the data, but the undersampling method produced better results with the F1 score metric. We decided to use this approach, but later in the model development process, we got to tuning the thresholds to optimize the F1 score for the models, and we noticed that the optimum threshold was always nearly default (0.5). Due to this, we decided to abandon the idea of trying to balance the data because we wanted to demonstrate our ability to properly implement threshold tuning in our models, which was something we learned in our data mining course. Plus, the ability to adjust the threshold allowed us to implement our domain knowledge and tune the model to perform better based on improving precision, recall, or both (F1 score).

2. **Threshold<a name="_page15_x72.00_y265.18"></a> Tuning**

Instead of balancing our data, we decided to fine-tune our models by adjusting the threshold to optimize the F1 score. In the instance of our Random Forest model, the best F1 score threshold for the model was (0.3289), which meant that the model's output probabilities for the positive class were skewed toward lower values. In other words, the model was more conservative in predicting the positive class and required higher evidence or confidence in the input data to make a positive prediction. This is due to a class imbalance in the target because the target (violence) is imbalanced 3:1 no-violence to violence, respectively. Since the positive class represents a rarer event, the model may be hesitant to predict it. Lowering the default threshold (0.5) to around (0.3) allowed the model to capture more positive instances at the cost of higher false positives. Adjusting the threshold allowed for the performance of the models to be improved on our primary metric of F1 score.

3. **Primary<a name="_page15_x72.00_y596.36"></a> Metric for Evaluation**

We knew that we needed a proper metric to assess the performance of our models. In the case of a binary classification problem that classifies whether someone is violent or not, optimizing for precision would mean minimizing the number of false positives. This would be appropriate if the cost of a false positive (i.e., classifying a non-violent person as violent) is high, such as in the case of taking legal action or enforcing a punishment. In this case, we would want to be very certain that the person being classified as violent is actually violent.

On the other hand, optimizing for recall would mean minimizing the number of false negatives. This would be appropriate if the cost of a false negative (i.e., failing to classify a violent person as violent) is high, such as in the case of public safety or preventing harm. In this case, we would want to identify as many violent individuals as possible, even if it means including some non-violent individuals in the classification.

Ultimately, we believed it was necessary to strike a balance between the two and optimize for a combination of precision and recall by using the F1 score as our primary metric to focus upon.

4. **Model<a name="_page16_x72.00_y375.57"></a> Development**

The general structure for all three models was implemented in a similar fashion. First, all three algorithms were implemented with all 125 features, which was our base model or “B” in the figure below. Then, the base model was tested with both feature selection methods Logistic Regression P-value and Random Forest feature importance. Since the P-value performed the best, we opted to use this method for the remainder of the model-building process. After feature selection was implemented, the models were then hyper-parameter-tuned by using a grid search and evaluated with an F1 score and a confusion matrix. Then, we adjusted the threshold for classifying with the purpose of optimizing the F1 score. Finally, the performance metrics of this final model were evaluated with F1 score, accuracy, ROC, AUC, and a confusion matrix. The figure below is a visual representation of the model development process.

**Figure 6**

*Model Development Process*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 006](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/d4baf039-ee82-465d-b5ef-a52c5982421a)


5. **Results**

<a name="_page17_x72.00_y324.49"></a>In this section, we present the results of our binary classification models using Logistic Regression, Decision Tree, and Random Forest algorithms. We first describe the rationale behind each model selection, present the results for each model, and compare their performance amongst all models.

The following metrics were used to assess the models' performance: F1-score, Accuracy, ROC-AUC, and Confusion Matrix. The primary metric utilized was F1-score, as stated in the methodology section of this paper.

1. **Logistic<a name="_page17_x72.00_y545.27"></a> Regression**

We selected Logistic Regression as one of our models because it is a commonly used algorithm for binary classification problems and works well with imbalanced datasets, and it is easy to interpret the results due to its parametric approach.

First, we created a base model with 125 features and tested its performance with F1 score (0.5083). Then, we applied the P-value feature selection, which reduced the features to 56, and again evaluated the performance with F1 score (0.4997). We observed that the model performed slightly worse with feature selection but was negligible (0.0086 worse). Next, we applied hyper-parameter tuning to the feature selection model by using a grid search. The figure below shows the best hyper-parameters provided by the grid search.

**Figure 7**

*Logistic Regression Parameters from Grid Search*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 007](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/c4b5d43a-c019-4a81-8ef4-7f10b4ffa764)


It is worth mentioning that the performance gain of hyper-parameter tuning was minimal over the base model (0.0008 improvement) based on the F1 score metric (0.5091). Finally, the model threshold was adjusted to optimize the F1 score (0.5985). The figure below shows the best-performing threshold.

**Figure 8**

*Logistic Regression Best F1-Score Threshold*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 008](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/780a5c69-71a8-406e-86d9-e2229bec55ed)


10

The table below shows the performance of Logistic Regression for each model development step with the F1 score metric and the threshold used for the final model. **Table 1**

*Logistic Regression F1-Score Performance on Test Data*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 009](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/a53f3ef7-f8f1-4485-acbe-1fd84f3073b5)


Threshold tuning to optimize the F1 score greatly boosted the model's ability to classify violence. The figures below depict the confusion matrix and ROC-AUC for the best-performing model (P-value + Parameter tuning + Threshold).

**Figure 9**

*Logistic Regression Best Performing Model Confusion Matrix*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 010](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/7d5ea979-be79-47f1-83d5-ab1d11968aae)


**Figure 10****
21

*Logistic Regression Best Performing Model ROC/AUC*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 011](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/645a414b-b6f3-4db0-a10d-7dfefbf12c44)

2. **Decision<a name="_page20_x72.00_y343.99"></a> Tree**

We selected a Decision Tree as one of our models because it is a simple and easily explainable model. Decision trees can be effective for imbalanced datasets because they can be easily tuned to improve the classification performance for the two classes.

First, we created a base model with all 125 features and tested its performance with F1 score (0.4559). Then, we applied P-value feature selection and again evaluated the performance with F1 score (0.4644). Unlike Logistic Regression, the Decision Tree performed better with feature selection by an F1 score (0.0105) difference. Next, we applied hyper-parameter tuning to the feature selection model by using a grid search. The figure below shows the best hyper-parameters provided by the tuning process.

**Figure 11**

*Decision Tree Parameters from Grid Search*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 012](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/b6575ebf-cb75-407f-9a64-a03c260e282b)



32

Notably, the performance gain of hyper-parameter tuning was significant over the base model by (0.0463) improvement based on the F1 score, which was not observed in the Logistic Regression model. Finally, the model threshold was adjusted to optimize the F1 score, which provided an even greater performance (0.5508). The figure below shows the best-performing threshold. It should be noted that the threshold was higher for the Decision Tree compared to Logistic Regression.

**Figure 12**

*Decision Tree Best F1-Score Threshold*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 013](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/65be1709-427e-4571-92a5-648d697cd428)


The table below shows the performance of the Decision Tree for each model development step with the F1 metric and the threshold used for the final model. **Table 2**

*Decision Tree F1-Score Performance on Test Data*


![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 014](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/d717a72b-8652-4d2e-8790-0c1b057602b5)

Hyper-parameter tuning and Threshold tuning to optimize the F1 score greatly boosted the model's ability to classify violence. The figures below depict the confusion matrix and ROC-AUC for the best-performing Decision Tree model (P-value + Parameter tuning + Threshold).

**Figure 13**

*Decision Tree Best Performing Model Confusion Matrix*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 015](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/4c85cd8d-6be7-4b25-a547-4b289960ebfe)


**Figure 14**

*Decision Tree Best Performing Model ROC/AUC*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 016](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/2a007cfa-9c01-44cf-a558-969a77db41d0)


3. **Random<a name="_page23_x72.00_y310.39"></a> Forest**

We selected Random Forest as one of our models because it uses an ensemble approach that handles imbalanced data well by randomly sampling data and features with replacement, which makes it less likely to overfit. Although, there is a trade-off to using this algorithm: higher accuracy but less interpretability.

First, we created a base model with all of the features and tested its performance with F1 score (0.5211). The base model performance for Random Forest was significantly higher than the prior two algorithm base models. Then, we applied P-value feature selection and again evaluated the performance with F1 score (0.5273). The model performed slightly better with feature selection, but the difference was negligible by a difference of (0.0062). Next, we applied hyper-parameter tuning to the feature selection model by using a grid search. The figure below shows the best hyper-parameters provided by the tuning process.

**Figure 15**

*Random Forest Parameters from Grid Search*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 017](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/0593d7f5-ab06-4bac-aa3f-54011a4842b3)


We compared the returned best parameters from our grid searches for the Decision Tree and Random Forest since they are both using a form of a Decision Tree. Both models tuned the same parameters: max\_depth, min\_samples\_leaf, and min\_samples\_split. They differed by the Decision Tree tuning the criterion and the Random Forest tuning the n\_estimators. Overall, the shared parameters were vastly different for the two models. For example, the Decision Tree had a max\_depth of 10, while the Random Forest had an unlimited max\_depth. Another example is the min\_samples\_leaf, where the Decision Tree had six, and the Random Forest had one. The reason that the shared parameters differed was due to the fact of how the Random Forest algorithm functions. Random Forest is an ensemble algorithm that makes use of bootstrap aggregation (Bagging), which means that it randomly samples (with replacement) features and data to construct numerous decision trees. This key difference is why the shared parameters are different for both models.

It is worth mentioning that the performance gain of hyper-parameter tuning for the Random Forest was minimal over the base model by (0.0102) improvement based on the F1 score. Finally, the model threshold was adjusted to optimize the F1 score (0.6123). The figure below shows the best-performing threshold. Note that the threshold for Random Forest was slightly lower than the Decision Tree but higher than the Logistic Regression threshold.

**Figure 16**

*Random Forest Best F1-Score Threshold*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 018](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/9cd77bda-ae6f-473f-807d-0f80de9cfab9)


The table below shows the performance of Random Forest for each model development step with the F1 score and the threshold used for the final model.

**Table 3**

*Random Forest F1-Score Performance on Test Data*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 019](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/3545cba1-1d54-42b8-9159-cb9fcc512d0f)


Threshold tuning to optimize the F1 score greatly boosted the model's ability to classify violence, the largest improvement of the three models. The figures below depict the confusion matrix and ROC-AUC for the best-performing model (P-value + Parameter tuning + Threshold). **Figure 17**

*Random Forest Best Performing Model Confusion Matrix*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 020](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/3c5994ad-eb17-4a5b-8d03-c1b54f525799)


**Figure 18**

*Random Forest Best Performing Model ROC/AUC*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 021](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/ef6d5b51-494b-43e5-9ab5-864da476454b)


4. **Model<a name="_page26_x72.00_y591.08"></a> Comparison**

A culmination of three metrics was employed to evaluate the performance of all three models. Among the set of metrics, the F1 score was designated as the primary measure of interest. The results indicated that the Random Forest model exhibited the highest performance on this metric, achieving an F1 score of 61.23%. This outperformed the next best model, Logistic Regression, by approximately 1.5%. In contrast, the Decision Tree model demonstrated the lowest F1 score, achieving only 55%. The Random Forest model further demonstrated superior performance on additional metrics, including accuracy and AUC. The Logistic Regression model secured the second-highest rank across all metrics, whereas the Decision Tree model exhibited the weakest overall performance. The table below summarizes model performance for F1 score, AUC, accuracy and includes the threshold implemented for the corresponding model.

**Table 4**

*All Models F1-Score, AUC, Accuracy, and Threshold Compared on Test Data*

![Aspose Words cd14414e-ca41-46f4-ba4e-161fe571d466 022](https://github.com/jaymonty/Predicting-Adolescent-Delinquency-Violence-Using-Data-Mining-Techniques/assets/18198506/dc26555d-673a-4e62-bf1d-44379d90b371)


6. **Discussion**

<a name="_page27_x72.00_y452.43"></a>In this section, we will discuss some of the key takeaways from the project, such as: why a particular model performed best, why some features are important, and the meaning of the results.

1. **Why<a name="_page27_x72.00_y562.82"></a> One Method is Better?**

The Random Forest model performs better than the Logistic Regression and Decision Tree models because it is a more sophisticated algorithm that is better suited for handling imbalanced datasets. In a Random Forest model, Decision Trees are constructed on different subsets of features and data. The classification is produced by the aggregation of the predictions from each tree. This can help to reduce the impact of outliers and noise in the data and also help to identify important features that may be predictive of the target variable. In addition, Random Forests are known to be effective in handling imbalanced datasets by providing balanced class weights to each Decision Tree. This can help to prevent the model from being biased towards the majority class, in which our dataset had an issue of three to one for the target feature violence. On the other hand, Logistic Regression and Decision Trees may struggle to handle imbalanced datasets as they do not inherently address the class imbalance. Logistic regression can fail to capture complex relationships between variables, while Decision Trees may suffer from overfitting or underfitting the data. Overall, the Random Forest model's ability to handle imbalanced datasets and capture complex relationships between variables defines why it outperformed the other models for accurately classifying adolescent violence.

2. **Why<a name="_page28_x72.00_y347.98"></a> Selected Features Important?**

Selected features are important because, in the Logistic Regression P-value feature selection, the method calculates the statistical significance of each feature with respect to the target variable. The P-value denotes the likelihood of the observed outcome (i.e., the association between a particular feature and the target variable violence) happening randomly. If the p-value is less than (0.01), the feature is considered statistically significant and selected for the model. Therefore, the 56 selected features are the ones that have a statistically significant relationship with the target variable (violence) and are the most informative for the classification task. By using only the selected features, the model can reduce overfitting, improve performance, and increase interpretability.

3. **Meaning<a name="_page28_x72.00_y623.95"></a> of Result**

The aim of this project was to develop a binary classification model that accurately         predicts the likelihood of violence among adolescents. The analysis was based on data collected from a large-scale survey of adolescents from 31 countries. The project's key findings indicate a strong correlation between violence and the lack of self-control and association with deviant friends.

The binary classification Random Forest model developed in this project accurately predicted the likelihood of violence among adolescents with a high degree of certainty; F1 score: 61.23%. The results showed that adolescents more likely to exhibit violent behavior are likelier to have friends who engage in deviant behavior and exhibit a lack of self-control.

The findings of this project are consistent with previous research that has identified the importance of self-control and peer associations in the development of violent behavior (Pauwels et al., 2011). The lack of self-control may lead to impulsive and aggressive behavior, while associating with deviant peers may normalize violent behavior and increase the likelihood of engaging in it. Furthermore, previous research has suggested that peer associations may provide social support and encouragement for violent behavior.

The findings of this project have important implications for preventing and intervening in violence among adolescents. Strategies that focus on developing self-control skills and promoting healthy peer associations may be effective in reducing the likelihood of violent behavior among adolescents. Additionally, identifying these risk factors may help educators, parents, and other professionals recognize and address early signs of violence.

In conclusion, this project provides evidence that violence among adolescents is strongly correlated with the lack of self-control and association with deviant friends. The findings suggest that prevention and intervention efforts should focus on developing self-control skills and promoting healthy peer associations to reduce the likelihood of violent behavior. Future research may further explore these risk factors and investigate additional factors that contribute to the development of violent behavior among adolescents.

<a name="_page30_x72.00_y529.60"></a>**References**

Chawla, N. V., Bowyer, K. W., Hall, L. J., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic

Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953

Enzmann, D., Marshall, I. H., Killias, M., Junger-Tas, J., Steketee, M., & Gruszczyńska, B.

(2010). Self-reported youth delinquency in Europe and beyond: First results of the Second International Self-Report Delinquency Study in the context of police and victimization data. *European Journal of Criminology*, *7*(2), 159–183. https://doi.org/10.1177/1477370809358018

Hosmer, D. W., & Lemeshow, S. (2000). Applied Logistic Regression. In *John Wiley & Sons,*

*Inc. eBooks*. https://doi.org/10.1002/0471722146

Louppe, G. (2014). Understanding Random Forests: From Theory to Practice. *arXiv*.

https://doi.org/10.13140/2.1.1570.5928

Pauwels, L., Vettenburg, N., Gavray, C., & Brondeel, R. (2011). Societal Vulnerability and

Troublesome Youth Group Involvement. *International Criminal Justice Review*, *21*(3), 283–296. https://doi.org/10.1177/1057567711419899

Posick, C. (2013). The Overlap Between Offending and Victimization Among Adolescents.

*Journal of Contemporary Criminal Justice*, *29*(1), 106–124. https://doi.org/10.1177/1043986212471250

Sabia, M. F. (2017). Predicting Self-Control Through Familial and Environmental Factors

Among Immigrant and Native-Born Adolescents. *Social Science Research Network*. https://doi.org/10.2139/ssrn.3013938

Sabia, M. F., Hickman, G. P., & Barkley, W. (2017). Predicting Delinquency through

Psychosocial and Environmental Variables among Immigrant and Native-Born

Adolescents. *Social Science Research Network*. https://doi.org/10.2139/ssrn.2994920 Stansfield, R. (2017). Teen Involvement in Sports and Risky Behaviour: A Cross-national and

Gendered Analysis. *British Journal of Criminology*. https://doi.org/10.1093/bjc/azv108


