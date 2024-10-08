package org.elvin;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.Random;

public class Main {
    public static void main(String[] args) {
        try {
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



            // Create and train NaiveBayes classifier
            Classifier naiveBayes = new NaiveBayesMultinomial();
            naiveBayes.buildClassifier(normalizedData);

            // Evaluate the model using cross-validation
            Evaluation evaluation = new Evaluation(normalizedData);
            evaluation.crossValidateModel(naiveBayes, normalizedData, 10, new Random(1));

            // Output evaluation results
            System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
            System.out.println("Confusion Matrix:");
            double[][] confusionMatrix = evaluation.confusionMatrix();
            for (double[] row : confusionMatrix) {
                for (double value : row) {
                    System.out.print(value + " ");
                }
                System.out.println();
            }

            // Classify a new instance (optional)
            // Assuming you have an Instances object for new data
            // Instances newData = ...;
            // double label = naiveBayes.classifyInstance(newData.instance(0));
            // System.out.println("Predicted class: " + filteredData.classAttribute().value((int) label));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

/*
* O Naive Bayes é uma escolha adequada para o problema dado, especialmente como um modelo inicial para avaliar a classificação entre Bart e Homer.
*  Contudo, se a precisão não for suficiente, pode ser interessante explorar modelos mais complexos que capturam interações entre atributos,
*  como árvores de decisão ou modelos baseados em redes neurais.
*  */