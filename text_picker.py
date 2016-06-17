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
        self.data = {str(key): value for key, value in data.items()}
        self.has_saved = False
        self.list_view = None
        self.list_model = None
        self.text_window = None
        self.save_button = None
        self.close_button = None
        self.current_selection = 0
        self.keys = list(self.data.keys())
        self.init_ui()
        self.set_selected(0)

    def init_ui(self):
        self.setWindowTitle('Filter texts.')
        layout = QtGui.QHBoxLayout()
        self.list_view = QtGui.QListView(self)
        self.list_model = TextModel(self.keys)
        self.list_view.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.list_view.setModel(self.list_model)
        self.list_view.selectionModel().selectionChanged.connect(
            self.list_changed
        )
        self.text_window = QtGui.QTextEdit(self)
        self.text_window.setReadOnly(True)
        layout.addWidget(self.list_view)
        layout.addWidget(self.text_window)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_button = QtGui.QPushButton('Save')
        self.save_button.clicked.connect(self.save)
        self.close_button = QtGui.QPushButton('Close')
        self.close_button.clicked.connect(self.close)

        self.base_string = '{{}} / {}'.format(len(self.data))
        self.label = QtGui.QLabel()
        sub_layout = QtGui.QVBoxLayout()
        sub_layout.addWidget(self.label)
        sub_layout.addStretch()
        sub_layout.addWidget(self.save_button)
        sub_layout.addWidget(self.close_button)
        layout.addLayout(sub_layout)
        self.setLayout(layout)

    def close(self):
        if not self.has_saved:
            question = ('You have not saved.\n\nWould you '
                        'like to save before closing?')
            answer = QtGui.QMessageBox().question(
                self, 'Save?', question,
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No |
                QtGui.QMessageBox.Abort)

            if answer == QtGui.QMessageBox.Yes:
                did_save = self.save()
                if not did_save:
                    return

            elif answer == QtGui.QMessageBox.Abort:
                return

        super(TextPickerWidget, self).close()

    def save(self):
        """ Open dialog and save filtered data to pickle-file.

        Returns
        -------
        bool
            True if progress was saved else False.
        """
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

            self.has_saved = True
            return True

        return False

    def list_changed(self, selection):
        if selection.indexes():
            i = selection.indexes()[0].row()
            self.change_display(i)

    def change_display(self, i):
        self.text_window.clear()
        self.current_selection = i
        self.text_window.setText(self.data[self.keys[i]])
        self.label.setText(self.base_string.format(i + 1))

    def set_selected(self, i):
        """

        Parameters
        ----------
        i : int
            Selected row.
        """
        if not (0 <= i < self.list_model.rowCount()):
            return

        qindex = self.list_model.createIndex(i, 0)
        sel_model = self.list_view.selectionModel()
        sel_model.clearSelection()
        sel_model.setCurrentIndex(qindex, QtGui.QItemSelectionModel.Select)

        self.change_display(i)

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
        self.has_saved = False
        is_good = self.list_model.is_good[self.current_selection]
        self.list_model.is_good[self.current_selection] = not is_good
        self.list_view.update()


class PickKeyDialog(QtGui.QDialog):

    def __init__(self, keys):
        super(PickKeyDialog, self).__init__()
        self.keys = keys
        self.buttons = list()
        self.button_group = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Pick data to filter.')
        layout = QtGui.QVBoxLayout()
        for key in self.keys:
            button = QtGui.QRadioButton(key)
            button.toggled.connect(self.enable_ok)
            self.buttons.append(button)
            layout.addWidget(button)

        self.button_group = QtGui.QDialogButtonBox(
            QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal
        )
        self.button_group.button(QtGui.QDialogButtonBox.Ok).clicked.connect(
            self.accept
        )
        self.button_group.button(QtGui.QDialogButtonBox.Cancel).clicked.connect(
            self.reject
        )
        self.button_group.button(QtGui.QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self.button_group)
        self.setLayout(layout)

    def get_values(self):
        checked_button = next(b for b in self.buttons if b.isChecked())
        return checked_button.text()

    def enable_ok(self, toggled):
        self.button_group.button(QtGui.QDialogButtonBox.Ok).setEnabled(True)



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    dialog = QtGui.QFileDialog()
    fname = dialog.getOpenFileName(filter='*.pickle')

    if fname:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        sys.exit()

    try:
        pick_dialog = PickKeyDialog(data.keys())
    except AttributeError:
        QtGui.QMessageBox().critical(None, 'Error', 'Wrong format.')
        sys.exit(1)

    if pick_dialog.exec_():
        key = pick_dialog.get_values()
    else:
        sys.exit()

    try:
        window = TextPickerWidget(data[key])
    except AttributeError:
        msg = 'Wrong format. Data must be key, value-pairs.'
        QtGui.QMessageBox().critical(None, 'Error', msg)
        sys.exit(1)

    window.show()
    sys.exit(app.exec_())