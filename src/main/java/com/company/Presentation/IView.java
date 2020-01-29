package com.company.Presentation;

public interface IView<TPresenter> {
  void setPresenter(TPresenter presenter);
  TPresenter getPresenter();

  void show();
}
