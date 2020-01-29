package com.company.View;


import com.company.Presentation.IApplicationController;
import com.company.Presentation.MainView.IMainViewPresenter;
import com.company.Presentation.MainView.MainViewPresenter;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;

public class ApplicationController implements IApplicationController {
  private Stage _mainStage;

  public ApplicationController(Stage mainStage){
    _mainStage = mainStage;
  }

  @Override
  public File showChooseImageView() {
    FileChooser fileChooser = new FileChooser();
    fileChooser.setTitle("Open Image File");
    fileChooser.setInitialDirectory(new File("E:\\projects\\back_end\\FaceMaven\\Temp"));
    File chosenFile = fileChooser.showOpenDialog(_mainStage);
    return chosenFile;
  }

  @Override
  public void showMainView() {
 com.company.View.MainView mainView = new MainView(_mainStage);
    IMainViewPresenter mainViewPresenter = new MainViewPresenter(mainView,this);

    mainView.show();
  }
}
