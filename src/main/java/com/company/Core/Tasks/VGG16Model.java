package com.company.Core.Tasks;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

/**
 * VGG-16, from Very Deep Convolutional Networks for Large-Scale Image Recognition
 * <a href="https://arxiv.org/abs/1409.1556">https://arxiv.org/abs/1409.1556</a><br>
 * <br>
 * Deep Face Recognition<br>
 * <a href="http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf">http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf</a>
 *
 * <p>ImageNet weights for this model are available and have been converted from <a href="https://github.com/fchollet/keras/tree/1.1.2/keras/applications">
 *     https://github.com/fchollet/keras/tree/1.1.2/keras/applications</a>.</p>
 * <p>CIFAR-10 weights for this model are available and have been converted using "approach 2" from <a href="https://github.com/rajatvikramsingh/cifar10-vgg16">
 *     https://github.com/rajatvikramsingh/cifar10-vgg16</a>.</p>
 * <p>VGGFace weights for this model are available and have been converted from <a href="https://github.com/rcmalli/keras-vggface">
 *     https://github.com/rcmalli/keras-vggface</a>.</p>
 *
 * @author Justin Long (crockpotveggies)
 */
@AllArgsConstructor
@Builder
public class VGG16Model extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 224, 224};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new Nesterovs();
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.SEPARATE;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    public VGG16Model() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return DL4JResources.getURLString("models/vgg16_dl4j_inference.zip");
        else if (pretrainedType == PretrainedType.CIFAR10)
            return DL4JResources.getURLString("models/vgg16_dl4j_cifar10_inference.v1.zip");
        else if (pretrainedType == PretrainedType.VGGFACE)
            return DL4JResources.getURLString("models/vgg16_dl4j_vggface_inference.v1.zip");
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return 3501732770L;
        if (pretrainedType == PretrainedType.CIFAR10)
            return 2192260131L;
        if (pretrainedType == PretrainedType.VGGFACE)
            return 2706403553L;
        else
            return 0L;
    }

    @Override
    public Model initPretrained(PretrainedType pretrainedType) throws IOException {
        String remoteUrl = this.pretrainedUrl(pretrainedType);
        if (remoteUrl == null) {
            throw new UnsupportedOperationException("Pretrained " + pretrainedType + " weights are not available for this model.");
        } else {
            String localFilename = (new File(remoteUrl)).getName();
            ROOT_CACHE_DIR.mkdirs();
            File cachedFile = new File(ROOT_CACHE_DIR.getAbsolutePath(), localFilename);
//            cachedFile.delete();
            if (!cachedFile.exists()) {
                System.out.println("Downloading model to " + cachedFile.toString());
                FileUtils.copyURLToFile(new URL(remoteUrl), cachedFile);
            } else {
                System.out.println("Using cached model at " + cachedFile.toString());
            }

            long expectedChecksum = this.pretrainedChecksum(pretrainedType);
            if (expectedChecksum != 0L) {
                System.out.println("Verifying download...");
                Checksum adler = new Adler32();
                FileUtils.checksum(cachedFile, adler);
                long localChecksum = adler.getValue();
                System.out.println("Checksum local is " + localChecksum + ", expecting " + expectedChecksum);
                if (expectedChecksum != localChecksum) {
                    System.out.println("Checksums do not match. Cleaning up files and failing...");
                    cachedFile.delete();
                    throw new IllegalStateException("Pretrained model file failed checksum. If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.");
                }
            }

            if (this.modelType() == MultiLayerNetwork.class) {
                return ModelSerializer.restoreMultiLayerNetwork(cachedFile);
            } else if (this.modelType() == ComputationGraph.class) {
                return ModelSerializer.restoreComputationGraph(cachedFile);
            } else {
                throw new UnsupportedOperationException("Pretrained models are only supported for MultiLayerNetwork and ComputationGraph.");
            }
        }
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    public ComputationGraphConfiguration conf() {
        ComputationGraphConfiguration conf =
                new NeuralNetConfiguration.Builder().seed(seed)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(updater)
                        .activation(Activation.RELU)
                        .cacheMode(cacheMode)
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .graphBuilder()
                        .addInputs("in")
                        // block 1
                        .addLayer(String.valueOf(0), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                .cudnnAlgoMode(cudnnAlgoMode).build(), "in")
                        .addLayer(String.valueOf(1), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(64).cudnnAlgoMode(cudnnAlgoMode).build(), "0")
                        .addLayer(String.valueOf(2), new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "1")
//                         block 2
                        .addLayer(String.valueOf(3), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "2")
                        .addLayer(String.valueOf(4), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build(), "3")
                        .addLayer(String.valueOf(5), new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "4")
                        // block 3
                        .addLayer(String.valueOf(6), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "5")
                        .addLayer(String.valueOf(7), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "6")
                        .addLayer(String.valueOf(8), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build(), "7")
                        .addLayer(String.valueOf(9), new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "8")
                        // block 4
                        .addLayer(String.valueOf(10), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "9")
                        .addLayer(String.valueOf(11), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "10")
                        .addLayer(String.valueOf(12), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "11")
                        .addLayer(String.valueOf(13), new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "12")
                        // block 5
                        .addLayer(String.valueOf(14), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "13")
                        .addLayer(String.valueOf(15), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "14")
                        .addLayer(String.valueOf(16), new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build(), "15")
                        .addLayer(String.valueOf(17), new SubsamplingLayer.Builder()
                                .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                .stride(2, 2).build(), "16")
                        .addLayer(String.valueOf(18), new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                                .build(), "17")
                        .addLayer(String.valueOf(19), new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                                .build(), "18")
                        .addLayer(String.valueOf(20), new OutputLayer.Builder(
                                LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                                .nOut(numClasses).activation(Activation.SOFTMAX) // radial basis function required
                                .build(), "19")
                        .setOutputs("20")
                        .setInputTypes(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                        .build();

        return conf;
    }

    @Override
    public ComputationGraph init() {
        ComputationGraph network = new ComputationGraph(conf());
        network.init();
        return network;
    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public ZooType zooType() {
        return ZooType.VGG16;
    }

//    @Override
//    public ZooType zooType() {
//        return ZooType.VGG16;
//    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

}