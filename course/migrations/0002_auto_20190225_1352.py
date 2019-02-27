# Generated by Django 2.1.7 on 2019-02-25 08:22

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('organization', '0003_auto_20190225_1352'),
        ('course', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Assignment',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_assign', models.BooleanField(default=False)),
                ('faculty', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='organization.Faculty')),
            ],
        ),
        migrations.RemoveField(
            model_name='subject',
            name='faculty',
        ),
        migrations.AddField(
            model_name='assignment',
            name='subject',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='course.Subject'),
        ),
        migrations.AddField(
            model_name='subject',
            name='faculty',
            field=models.ManyToManyField(through='course.Assignment', to='organization.Faculty'),
        ),
    ]