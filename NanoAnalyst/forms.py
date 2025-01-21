# uploader/forms.py

from django import forms


class UploadFileForm(forms.Form):
    file = forms.FileField(label='Select an Excel file')

    # Plot 1 Range Inputs
    plot1_xmin = forms.FloatField(required=False, label='Plot 1 X-axis Minimum')
    plot1_xmax = forms.FloatField(required=False, label='Plot 1 X-axis Maximum')
    plot1_ymin = forms.FloatField(required=False, label='Plot 1 Y-axis Minimum')
    plot1_ymax = forms.FloatField(required=False, label='Plot 1 Y-axis Maximum')

    # Plot 2 Range Inputs
    plot2_xmin = forms.FloatField(required=False, label='Plot 2 X-axis Minimum')
    plot2_xmax = forms.FloatField(required=False, label='Plot 2 X-axis Maximum')
    plot2_ymin = forms.FloatField(required=False, label='Plot 2 Y-axis Minimum')
    plot2_ymax = forms.FloatField(required=False, label='Plot 2 Y-axis Maximum')

    def clean_file(self):
        file = self.cleaned_data.get('file', False)
        if file:
            if not file.name.endswith(('.xls', '.xlsx')):
                raise forms.ValidationError("Unsupported file type. Please upload an Excel file.")
            if file.size > 10 * 1024 * 1024:
                raise forms.ValidationError("File too large. Size should not exceed 10 MB.")
            return file
        else:
            raise forms.ValidationError("Couldn't read uploaded file.")

    def clean(self):
        cleaned_data = super().clean()

        # Validate Plot 1 Ranges
        plot1_xmin = cleaned_data.get('plot1_xmin')
        plot1_xmax = cleaned_data.get('plot1_xmax')
        plot1_ymin = cleaned_data.get('plot1_ymin')
        plot1_ymax = cleaned_data.get('plot1_ymax')

        if plot1_xmin is not None and plot1_xmax is not None:
            if plot1_xmin >= plot1_xmax:
                self.add_error('plot1_xmin', 'Plot 1 X-axis Minimum must be less than Maximum.')

        if plot1_ymin is not None and plot1_ymax is not None:
            if plot1_ymin >= plot1_ymax:
                self.add_error('plot1_ymin', 'Plot 1 Y-axis Minimum must be less than Maximum.')

        # Validate Plot 2 Ranges
        plot2_xmin = cleaned_data.get('plot2_xmin')
        plot2_xmax = cleaned_data.get('plot2_xmax')
        plot2_ymin = cleaned_data.get('plot2_ymin')
        plot2_ymax = cleaned_data.get('plot2_ymax')

        if plot2_xmin is not None and plot2_xmax is not None:
            if plot2_xmin >= plot2_xmax:
                self.add_error('plot2_xmin', 'Plot 2 X-axis Minimum must be less than Maximum.')

        if plot2_ymin is not None and plot2_ymax is not None:
            if plot2_ymin >= plot2_ymax:
                self.add_error('plot2_ymin', 'Plot 2 Y-axis Minimum must be less than Maximum.')


# uploader/forms.py

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True, label='Email Address')

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


# uploader/forms.py

class AxesSelectionForm(forms.Form):
    x_axis = forms.ChoiceField(label="Select X-axis", choices=[], widget=forms.Select(attrs={'class': 'form-select'}))
    y_axis = forms.ChoiceField(label="Select Y-axis", choices=[], widget=forms.Select(attrs={'class': 'form-select'}))

    def __init__(self, *args, **kwargs):
        headers = kwargs.pop('headers', [])
        super(AxesSelectionForm, self).__init__(*args, **kwargs)
        choices = [('', '---------')] + [(header, header) for header in headers]
        self.fields['x_axis'].choices = choices
        self.fields['y_axis'].choices = choices
