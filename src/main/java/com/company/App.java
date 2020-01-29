package com.company;

import com.company.Presentation.IApplicationController;
import com.company.View.ApplicationController;
import javafx.application.Application;
import javafx.stage.Stage;

public class App extends Application {
  @Override
  public void start(Stage primaryStage) throws Exception {
    IApplicationController applicationController = new ApplicationController(primaryStage);
    applicationController.showMainView();
  }
}
