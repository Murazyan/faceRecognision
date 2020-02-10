package com.company.Presentation.MainView;


import com.company.Core.Math.EuclideanDistance;
import com.company.Core.Tasks.CreateInputVectorTask;
import com.company.Core.Tasks.ExtractFacesFromImageTask;
import com.company.Presentation.IApplicationController;
import com.company.View.Controls.Events.IOnRecognizeEventListener;
import com.company.View.Controls.Events.IOnTrainEventListener;
import com.company.View.Controls.FaceControlView;
import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Size;

import java.io.File;
import java.util.ArrayList;
import java.util.List;



public class MainViewPresenter implements IMainViewPresenter, IOnTrainEventListener, IOnRecognizeEventListener {
  private final IMainView _view;
  private final IApplicationController _applicationController;

  private SimpleStringProperty _message;

  private List<company.Presentation.Models.PersonModel> _trainList;

  private CreateInputVectorTask _createInputVector;

  private EuclideanDistance _distance;

  public MainViewPresenter(IMainView view, IApplicationController applicationController){
    _view = view;
    _view.setPresenter(this);
    _applicationController = applicationController;

    _createInputVector =new CreateInputVectorTask();

    _trainList = new ArrayList<>();

    _distance = new EuclideanDistance();

    _message = new SimpleStringProperty();
    _message.addListener(new ChangeListener<String>() {
      @Override
      public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
        System.out.println("Extracted Face Location: " + newValue);
        if (newValue.trim().length()==0)
          return;

        FaceControlView faceControlView = new FaceControlView();
        faceControlView.setImage(new File(newValue));
        faceControlView.addOnTrainListener((IOnTrainEventListener)_view.getPresenter());
        faceControlView.addOnRecognizeListener((IOnRecognizeEventListener)_view.getPresenter());

        Platform.runLater(new Runnable() {
          @Override public void run() {
            _view.addFaceControlView(faceControlView);
          }
        });

      }
    });
  }

  @Override
  public IMainView getView() {
    return _view;
  }

  @Override
  public void extractFaces(File imageFilePath) {
    _view.clearFaces();
    ExtractFacesFromImageTask extractFacesFromImageTask = new ExtractFacesFromImageTask("Extract Faces",_applicationController,
        imageFilePath,new File("E:\\projects\\back_end\\FaceMaven\\Temp\\cascade.xml"),new File("E:\\projects\\back_end\\FaceMaven\\Temp"),
        1.05,7,new Size(10,10), new Size(200,200));

    _message.bind(extractFacesFromImageTask.messageProperty());

    new Thread(extractFacesFromImageTask).start();
  }

  @Override
  public File showFileDialogChooser() {
    return _applicationController.showChooseImageView();
  }

  @Override
  public void onTrain(String personName, File faceImageFile) {
    System.out.println("Train: " + faceImageFile.getName());

      double[] faceFeatureArray = _createInputVector.runTask(faceImageFile);
      company.Presentation.Models.PersonModel objPersonModel = new company.Presentation.Models.PersonModel(personName,faceFeatureArray);
      _trainList.add(objPersonModel);
  }

  @Override
  public void onRecognize(FaceControlView sender, File personImageFile) {
    System.out.println("Recognize: " + personImageFile.getName());
    double[] faceFeatureArray = _createInputVector.runTask(personImageFile);
    INDArray array1 = Nd4j.create(faceFeatureArray);

    double minimalDistance = Double.MAX_VALUE;
    String result = "";
    for(company.Presentation.Models.PersonModel personModel : _trainList)
    {
      INDArray array2 = Nd4j.create(personModel.get_faceFeatureArray());
      double distance = _distance.run(array1,array2);
      if (distance<minimalDistance){
        minimalDistance = distance;
        result = personModel.get_personName();
      }
    }

    sender.setPersonName(result);
  }
}
