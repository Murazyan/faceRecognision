package com.company.Presentation.MainView;

import com.company.Presentation.IPresenter;

import java.io.File;

public interface IMainViewPresenter extends IPresenter<IMainView> {
  void extractFaces(File imageFilePath);
  File showFileDialogChooser();
}
