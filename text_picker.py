""" This module contains a script to quickly screen text data
for outliers.
"""
from PyQt4 import QtGui, QtCore
import pickle
import sys


class TextModel(QtGui.QStringListModel):

    def __init__(self, strings, *args, **kwargs):
        super(TextModel, self).__init__(strings, *args, **kwargs)
        self.is_good = [True for _ in strings]

    def data(self, index, role):
        if role == QtCore.Qt.BackgroundColorRole:
            if not self.is_good[index.row()]:
                return QtGui.QColor(QtCore.Qt.red)

        return super(TextModel, self).data(index, role)


class TextPickerWidget(QtGui.QWidget):

    """ This widget takes a dictionary of text bodies
    and displays them one by one.
    """

    def __init__(self, data, *args, **kwargs):
        super(TextPickerWidget, self).__init__(*args, **kwargs)
        self.data = data
        self.current_selection = 0

        layout = QtGui.QHBoxLayout()
        self.list_view = QtGui.QListView(self)

        self.keys = list(data.keys())
        self.list_model = TextModel(self.keys)
        self.list_view.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.list_view.setModel(self.list_model)
        self.list_view.selectionModel().selectionChanged.connect(
            self.list_changed
        )

        self.text_window = QtGui.QTextEdit(self)
        self.text_window.setReadOnly(True)

        self.set_selected(0)

        layout.addWidget(self.list_view)
        layout.addWidget(self.text_window)

        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def save(self):

        to_save = dict()
        for i, key in enumerate(self.keys):
            if self.list_model.is_good[i]:
                to_save[key] = self.data[key]

        dialog = QtGui.QFileDialog()
        fname = dialog.getSaveFileName(self, 'Save file',
                                       filter='*.pickle')
        if fname:
            with open(fname, 'wb') as f:
                pickle.dump(to_save, f)

    def list_changed(self, selection):
        if selection.indexes():
            self.text_window.clear()
            i = selection.indexes()[0].row()
            self.current_selection = i
            self.text_window.setText(self.data[self.keys[i]])

    def set_selected(self, i):
        """

        Parameters
        ----------
        i : int
            Selected row.
        """
        if not (0 <= i < self.list_model.rowCount()):
            return

        self.current_selection = i
        qindex = self.list_model.createIndex(i, 0)
        sel_model = self.list_view.selectionModel()
        sel_model.clearSelection()
        sel_model.setCurrentIndex(qindex, QtGui.QItemSelectionModel.Select)

        self.text_window.clear()
        self.text_window.setText(self.data[self.keys[i]])

    def keyPressEvent(self, event):
        """ On right-key press go to next.

        Parameters
        ----------
        event : QtGui.QKeyEvent
            Triggered event.
        """
        if event.key() == QtCore.Qt.Key_Right:
            self.go_to_next()
        elif event.key() == QtCore.Qt.Key_Left:
            self.go_back()
        elif event.key() in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return):
            self.toggle_bad()

    def go_to_next(self):
        self.set_selected(self.current_selection + 1)

    def go_back(self):
        self.set_selected(self.current_selection - 1)

    def toggle_bad(self):
        is_good = self.list_model.is_good[self.current_selection]
        self.list_model.is_good[self.current_selection] = not is_good
        self.list_view.update()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    dialog = QtGui.QFileDialog()
    fname = dialog.getOpenFileName(filter='*.pickle')

    if fname:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        sys.exit()

    window = TextPickerWidget(data)
    window.show()

    sys.exit(app.exec_())