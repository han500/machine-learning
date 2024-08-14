from django import forms

class InputForm(forms.Form):
    image = forms.ImageField()