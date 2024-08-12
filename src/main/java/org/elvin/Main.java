package org.elvin;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class Main {
    public static void main(String[] args) {
        try {
            DataSource source = new DataSource(Main.class.getResourceAsStream("/text/caracteristicas.arff"));
            Instances dataset = source.getDataSet();

            dataset.setClassIndex(dataset.numAttributes() - 1);

            Classifier classifier = new NaiveBayes();
            classifier.buildClassifier(dataset);

            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(classifier, dataset, 10, new Random(1));

            System.out.println(eval.toSummaryString("\nResultados da Avaliação\n", false));
            System.out.println("Matriz de Confusão: ");
            double[][] confusionMatrix = eval.confusionMatrix();
            for (double[] matrix : confusionMatrix) {
                for (double v : matrix) {
                    System.out.print(v + " ");
                }
                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}