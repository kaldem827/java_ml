package org.elvin;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.Random;

public class Test {

        public static void main(String[] args) throws Exception {
            DataSource source = new DataSource(Main.class.getResourceAsStream("/text/caracteristicas.arff"));
            Instances dataset = source.getDataSet();


            // Setting the class attribute (last attribute)
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }

            // Handle missing values by replacing them with the mean/mode
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(dataset);
            Instances dataNoMissing = Filter.useFilter(dataset, replaceMissing);

            // Preprocessing: Normalize the data (important for Naive Bayes when features are on different scales)
            Normalize normalize = new Normalize();
            normalize.setInputFormat(dataNoMissing);
            Instances normalizedData = Filter.useFilter(dataNoMissing, normalize);

            // Shuffle the data to ensure random distribution of instances
            normalizedData.randomize(new Random(1));

            // Split the dataset into 80% training and 20% testing
            int trainSize = (int) Math.round(normalizedData.numInstances() * 0.8);
            int testSize = normalizedData.numInstances() - trainSize;
            Instances trainData = new Instances(normalizedData, 0, trainSize);
            Instances testData = new Instances(normalizedData, trainSize, testSize);

            NaiveBayes naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(trainData);

            Evaluation evaluation = new Evaluation(trainData);
            evaluation.evaluateModel(naiveBayes, testData);

            System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
            System.out.println("Confusion Matrix:");
            double[][] confusionMatrix = evaluation.confusionMatrix();
            for (double[] row : confusionMatrix) {
                for (double value : row) {
                    System.out.print(value + " ");
                }
                System.out.println();
            }


        }
}
