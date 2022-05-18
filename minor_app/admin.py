from django.contrib import admin
from django.db.models.query_utils import RegisterLookupMixin
from minor_app.models import contact
# Register your models here.
admin.site.register(contact)
