package com.company.View.Controls.Events;

import com.company.View.Controls.FaceControlView;

import java.io.File;

public interface IOnRecognizeEventListener {
  void onRecognize(FaceControlView sender, File personImageFile);
}
