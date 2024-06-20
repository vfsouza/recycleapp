from django import forms


class RecycleForm(forms.Form):
    image = forms.ImageField(label='Image', required=True)
