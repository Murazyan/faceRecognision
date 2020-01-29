package com.company.Presentation.MainView;

import com.company.Presentation.IView;
import com.company.View.Controls.FaceControlView;

public interface IMainView extends IView<IMainViewPresenter> {
  void addFaceControlView(FaceControlView faceControlView);
  void clearFaces();
}
