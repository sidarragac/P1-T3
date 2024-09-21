from django.shortcuts import render

def recommendations(request):
    return render(request, 'recommendations.html')