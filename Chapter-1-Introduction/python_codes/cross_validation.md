# Cross valicdation
## Reason
���������ǽ����ݼ��ֳ�S��T�ֱ�����ѵ���Ͳ��ԣ�����ʱ�������������������⣬���Լ�����Ϣ���Ե߸���ѵ���õ�ģ�ͣ����ǹ���ϵ������Ϊ����������⣬����Ӧ��׼��һ�������ݼ�-��֤����ʹģ��ѵ����ɺ󣬶�ģ�ͽ���������������ڲ��Լ��Ͻ���������Ӧ�ý�����֤���ԣ�cv�����н����
## Defination
������֤�����Ȱ����ݼ��ֳ�k����С���ƵĻ����Ӽ�����ͨ���ֲ�����Ա�֤���ݷֲ�һ�£�Ȼ����k-1���Ӽ��Ĳ�����Ϊѵ���������µ��Ӽ���Ϊ���Լ����������Ի��k��ѵ��/���Լ��ϣ�����k��ѵ���Ͳ��ԣ�����k������ľ�ֵ��
## Thinking
�������� --> cv��ָ��(scores) --> ������֤������ --> Ӧ��
## ���㽻����֤��ָ��(scores)
* k-�۽�����أ�cross_val_score(clf, iris.data, iris.target, cv=k)
* �ı�scoring�ļ��㷽ʽ��scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring = 'f1_macro')
	* scoring ����: ����ģ����������(��[�ع飬���࣬����](http://sklearn.apachecn.org/cn/0.19.0/modules/model_evaluation.html#scoring-parameter)֮��)
* ����������֤���ԣ�
	* cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
	* cross_val_score(clf, iris.data, iris.target, cv = cv)
* ����Ԥ����(��׼��,��ֵȥ���Ͱ������������):
	* scaler = preprocessing.StandardScaler().fit(X_train)
	* X_train_transformed = scaler.transform(X_train)
	* X_test_transformed = scaler.transform(X_test)
* cross_validate �����Ͷ��������:
	* scoring = ['precision_macro', 'recall_macro']
	* scoring = {'prec_macro': 'precision_macro', 'rec_micro': make_scorer(recall_score, average='macro')}
* ͨ��������֤��ȡԤ��:
	* predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
	* metrics.accuracy_score(iris.target, predicted) 
## ������֤������
��Բ�ͬ���������ͣ�����ѡ�ò�ͬ�Ľ�����֤���������д�����Ҫ�������¼������棺
   1. ���IID�������ݣ�
 	* k�ۡ��ظ� K-�۽�����֤����һ������֤(LOO)����P������֤(LPO)��������н�����֤
   2. ��Ŀ�����ķֲ��Ͽ��ܱ��ֳ��ܴ�Ĳ�ƽ���ԣ����磬���ܻ���ֱ��������������ĸ�������
       * �ֲ�k�ۡ� �ֲ���� Split
   3. �����ķֲ�����������groups�����ݣ�����Ӷ�������ռ�ҽѧ���ݣ���ÿ���������ϲɼ�������������������ݺܿ���ȡ���ڸ���Ⱥ��
       # ��k-flod����һ�齻����֤����p�齻����֤��Group Shuffle Split



























