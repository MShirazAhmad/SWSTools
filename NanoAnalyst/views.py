# uploader/views.py

import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from .forms import UploadFileForm, RegisterForm
from .models import UploadedFile, GeneratedPlot
from django.contrib.auth import login, authenticate
from django.contrib.auth.models import User

# Registration View
def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('dashboard')
    else:
        form = RegisterForm()
    return render(request, 'registration/registration.html', {'form': form})

# Dashboard View
@method_decorator(login_required, name='dispatch')
class DashboardView(View):
    template_name = 'registration/dashboard.html'

    def get(self, request):
        user = request.user
        uploaded_files = UploadedFile.objects.filter(user=user).order_by('-uploaded_at')
        context = {
            'uploaded_files': uploaded_files,
        }
        return render(request, self.template_name, context)

# File Upload and Plot Generation View
@method_decorator(login_required, name='dispatch')
class FileUploadView(View):
    form_class = UploadFileForm
    template_name = 'NanoAnalyst/base.html'  # Ensure this path is correct

    def get(self, request):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']

            # Save the UploadedFile instance
            uploaded_instance = UploadedFile.objects.create(
                user=request.user,
                file=uploaded_file
            )

            # Retrieve plot range inputs for Plot 1
            plot1_xmin = form.cleaned_data.get('plot1_xmin')
            plot1_xmax = form.cleaned_data.get('plot1_xmax')
            plot1_ymin = form.cleaned_data.get('plot1_ymin')
            plot1_ymax = form.cleaned_data.get('plot1_ymax')

            # Retrieve plot range inputs for Plot 2
            plot2_xmin = form.cleaned_data.get('plot2_xmin')
            plot2_xmax = form.cleaned_data.get('plot2_xmax')
            plot2_ymin = form.cleaned_data.get('plot2_ymin')
            plot2_ymax = form.cleaned_data.get('plot2_ymax')

            # Process the file and generate plots
            try:
                combined_df = self.process_excel(uploaded_instance.file)
                plot1 = self.generate_plot1(
                    combined_df,
                    xmin=plot1_xmin,
                    xmax=plot1_xmax,
                    ymin=plot1_ymin,
                    ymax=plot1_ymax
                )
                plot2 = self.generate_plot2(
                    combined_df,
                    xmin=plot2_xmin,
                    xmax=plot2_xmax,
                    ymin=plot2_ymin,
                    ymax=plot2_ymax
                )

                # Save Plot 1 to in-memory file
                plot1_io = io.BytesIO()
                plot1.savefig(plot1_io, format='png')
                plot1_io.seek(0)
                plot1_base64 = base64.b64encode(plot1_io.getvalue()).decode('utf-8')

                # Save Plot 2 to in-memory file
                plot2_io = io.BytesIO()
                plot2.savefig(plot2_io, format='png')
                plot2_io.seek(0)
                plot2_base64 = base64.b64encode(plot2_io.getvalue()).decode('utf-8')

                # Save plots to media folder and associate with UploadedFile
                plot1_filename = default_storage.save(
                    f'plots/{uploaded_instance.id}_plot1.png',
                    ContentFile(plot1_io.getvalue())
                )
                plot2_filename = default_storage.save(
                    f'plots/{uploaded_instance.id}_plot2.png',
                    ContentFile(plot2_io.getvalue())
                )

                # Create GeneratedPlot instances
                generated_plot1 = GeneratedPlot.objects.create(
                    uploaded_file=uploaded_instance,
                    plot=plot1_filename
                )
                generated_plot2 = GeneratedPlot.objects.create(
                    uploaded_file=uploaded_instance,
                    plot=plot2_filename
                )

                context = {
                    'form': form,
                    'plot1': plot1_base64,
                    'plot2': plot2_base64,
                    'plot1_url': generated_plot1.plot.url,
                    'plot2_url': generated_plot2.plot.url,
                    'plot1_ranges': {
                        'xmin': plot1_xmin,
                        'xmax': plot1_xmax,
                        'ymin': plot1_ymin,
                        'ymax': plot1_ymax,
                    },
                    'plot2_ranges': {
                        'xmin': plot2_xmin,
                        'xmax': plot2_xmax,
                        'ymin': plot2_ymin,
                        'ymax': plot2_ymax,
                    },
                }
                return render(request, self.template_name, context)
            except Exception as e:
                form.add_error(None, f"Error processing file: {e}")

        return render(request, self.template_name, {'form': form})

    def process_excel(self, file):
        """
        Processes the uploaded Excel file and combines data from specified sheets.

        Args:
            file: Uploaded Excel file.

        Returns:
            Combined pandas DataFrame containing data from all valid sheets.
        """
        sheet_names = [f"Test {str(i).zfill(3)}" for i in range(1, 21)]
        combined_df = pd.DataFrame()

        for i, sheet_name in enumerate(sheet_names, start=1):
            try:
                # Read the Excel file from the uploaded file's path
                df_loaded = pd.read_excel(
                    file,
                    sheet_name=sheet_name,
                    dtype=str,
                    engine='xlrd'
                )

                # Rename columns
                df_loaded.columns = [
                    'Segment',
                    f'Displacement Into Surface (nm) - {i}',
                    f'Load On Sample (mN) - {i}',
                    f'Time On Sample (s) - {i}',
                    f'Harmonic Contact Stiffness (N/m) - {i}',
                    f'Hardness (GPa) - {i}',
                    f'Modulus (GPa) - {i}'
                ]

                # Drop the 'Segment' column
                df_loaded = df_loaded.drop(columns=['Segment'])

                # Combine with the main DataFrame
                if combined_df.empty:
                    combined_df = df_loaded
                else:
                    combined_df = pd.concat([combined_df, df_loaded], axis=1)
            except Exception as e:
                # Log the error or handle it as needed
                print(f"Error processing sheet {sheet_name}: {e}")
                pass
        return combined_df

    def generate_plot1(self, combined_df, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Generates Plot 1: Load vs Hardness Across Sheets.

        Args:
            combined_df: Combined pandas DataFrame containing data from all sheets.
            xmin: Minimum value for the X-axis.
            xmax: Maximum value for the X-axis.
            ymin: Minimum value for the Y-axis.
            ymax: Maximum value for the Y-axis.

        Returns:
            Matplotlib figure object.
        """
        # Extract all "Load On Sample (mN)" and "Hardness (GPa)" columns
        displacement_columns = [col for col in combined_df.columns if "Displacement Into Surface (nm)" in col and "Tagged" not in col]
        hardness_columns = [col for col in combined_df.columns if "Hardness (GPa)" in col and "Tagged" not in col]

        # Ensure the number of columns match
        if len(displacement_columns) != len(hardness_columns):
            raise ValueError("Mismatch in number of load and hardness columns after filtering.")

        # Initialize the plot using object-oriented interface
        fig, ax = plt.subplots(figsize=(10, 6))

        for load_col, displacement_col in zip(displacement_columns, hardness_columns):
            try:
                # Convert to numeric, skipping invalid entries
                load_data = pd.to_numeric(combined_df[load_col], errors='coerce')
                displacement_data = pd.to_numeric(combined_df[displacement_col], errors='coerce')

                # Plot valid data
                sheet_number = load_col.split('-')[-1].strip()
                ax.plot(load_data, displacement_data, label=f"Test {sheet_number}")  # Use sheet number in legend

                # Apply y-axis limits if provided
                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)
            except Exception as e:
                print(f"Error processing {load_col} and {displacement_col}: {e}")
                pass

        # Apply x-axis limits if provided
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)

        # Customize the plot
        ax.set_title("Load vs Hardness Across Sheets", fontsize=14)
        ax.set_xlabel("Displacement Into Surface (nm)", fontsize=12)
        ax.set_ylabel("Hardness (GPa)", fontsize=12)
        ax.legend(title="Sheet Number", loc="best")
        ax.grid(True)
        fig.tight_layout()

        return fig

    def generate_plot2(self, combined_df, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Generates Plot 2: Average Displacement vs Hardness with Standard Deviation Shading.

        Args:
            combined_df: Combined pandas DataFrame containing data from all sheets.
            xmin: Minimum value for the X-axis.
            xmax: Maximum value for the X-axis.
            ymin: Minimum value for the Y-axis.
            ymax: Maximum value for the Y-axis.

        Returns:
            Matplotlib figure object.
        """
        # Extract all "Displacement Into Surface (nm)" and "Hardness (GPa)" columns
        displacement_columns = [col for col in combined_df.columns if
                                "Displacement Into Surface (nm)" in col and "Tagged" not in col]
        hardness_columns = [col for col in combined_df.columns if "Hardness (GPa)" in col and "Tagged" not in col]

        if len(displacement_columns) != len(hardness_columns):
            raise ValueError("Mismatch in number of displacement and hardness columns after filtering.")

        all_displacements = []
        all_hardness = []

        for disp_col, hardness_col in zip(displacement_columns, hardness_columns):
            try:
                displacement_data = pd.to_numeric(combined_df[disp_col], errors='coerce').dropna()
                hardness_data = pd.to_numeric(combined_df[hardness_col], errors='coerce').dropna()

                all_displacements.append(displacement_data.values)
                all_hardness.append(hardness_data.values)
            except Exception as e:
                print(f"Error processing {disp_col} and {hardness_col}: {e}")
                pass

        if not all_displacements or not all_hardness:
            raise ValueError("No valid data found for plotting.")

        # Ensure all arrays are of the same length
        min_length = min(map(len, all_displacements + all_hardness))
        aligned_displacements = np.array([data[:min_length] for data in all_displacements])
        aligned_hardness = np.array([data[:min_length] for data in all_hardness])

        # Calculate averages and standard deviations
        avg_displacements = np.mean(aligned_displacements, axis=0)
        std_displacements = np.std(aligned_displacements, axis=0)
        avg_hardness = np.mean(aligned_hardness, axis=0)
        std_hardness = np.std(aligned_hardness, axis=0)

        # Initialize the plot using object-oriented interface
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(avg_displacements, avg_hardness, label="Average", color="green", linewidth=2)
        ax.fill_between(
            avg_displacements,
            avg_hardness - std_hardness,
            avg_hardness + std_hardness,
            color="green",
            alpha=0.2,
            label="± Standard Deviation"
        )

        # Apply plot ranges if provided
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)

        # Customize the plot
        ax.set_title("Average Displacement vs Hardness with Standard Deviation Shading", fontsize=14)
        ax.set_xlabel("Displacement Into Surface (nm)", fontsize=12)
        ax.set_ylabel("Hardness (GPa)", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

        return fig


# uploader/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .forms import *
from .models import UploadedFile, GeneratedPlot
import pandas as pd
import matplotlib.pyplot as plt
import os
from django.conf import settings
from django.core.files.base import ContentFile
import io
from django.contrib import messages


@login_required
def axes_selection_view(request, file_id):
    """
    Handle axis selection for generating plots from an uploaded Excel file.

    Args:
        request: The HTTP request object.
        file_id: The ID of the uploaded file.

    Returns:
        An HTTP response with the axis selection form or redirects to the dashboard after processing.
    """
    # Retrieve the uploaded file; ensure it belongs to the current user
    uploaded_file = get_object_or_404(UploadedFile, id=file_id, user=request.user)

    if request.method == 'POST':
        # Read the Excel file to extract headers
        try:
            excel_file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.file.name)
            # Read all sheets
            df = pd.read_excel(excel_file_path, sheet_name=None)
            all_headers = []
            for sheet_name, sheet_df in df.items():
                headers = list(sheet_df.columns)
                all_headers.extend(headers)
            # Remove duplicates while preserving order
            unique_headers = list(dict.fromkeys(all_headers))
        except Exception as e:
            messages.error(request, f"Error processing Excel file: {e}")
            return redirect('dashboard')

        # Bind form with POST data
        form = AxesSelectionForm(request.POST, headers=unique_headers)
        if form.is_valid():
            x_axis = form.cleaned_data['x_axis']
            y_axis = form.cleaned_data['y_axis']

            # Check if X and Y axes are selected
            if not x_axis or not y_axis:
                messages.error(request, "Both X-axis and Y-axis must be selected.")
                return render(request, 'uploader/select_axes.html', {'form': form, 'file': uploaded_file})

            # Check if X and Y axes are different
            if x_axis == y_axis:
                messages.error(request, "X-axis and Y-axis cannot be the same.")
                return render(request, 'uploader/select_axes.html', {'form': form, 'file': uploaded_file})

            # Generate plots based on selected axes
            try:
                for sheet_name, sheet_df in df.items():
                    # Ensure selected axes exist in the current sheet
                    if x_axis in sheet_df.columns and y_axis in sheet_df.columns:
                        plt.figure(figsize=(8, 6))
                        plt.scatter(sheet_df[x_axis], sheet_df[y_axis], alpha=0.7, edgecolors='w', s=100)
                        plt.title(f'{y_axis} vs {x_axis} - {sheet_name}')
                        plt.xlabel(x_axis)
                        plt.ylabel(y_axis)
                        plt.grid(True, linestyle='--', alpha=0.5)

                        # Save plot to in-memory file
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        plt.close()
                        buf.seek(0)
                        image_file = ContentFile(buf.read(), f'plot_{sheet_name}.png')

                        # Save GeneratedPlot instance
                        plot_instance = GeneratedPlot.objects.create(
                            uploaded_file=uploaded_file,
                            x_axis=x_axis,
                            y_axis=y_axis
                        )
                        plot_instance.plot_image.save(f'plot_{sheet_name}.png', image_file)
                messages.success(request, 'Plots generated successfully!')
            except Exception as e:
                messages.error(request, f"Error generating plots: {e}")
                return redirect('dashboard')

            return redirect('dashboard')
        else:
            # Form is invalid; re-render the form with errors
            return render(request, 'uploader/select_axes.html', {'form': form, 'file': uploaded_file})
    else:
        # Handle GET request: Display the axes selection form
        try:
            excel_file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.file.name)
            # Read all sheets
            df = pd.read_excel(excel_file_path, sheet_name=None)
            all_headers = []
            for sheet_name, sheet_df in df.items():
                headers = list(sheet_df.columns)
                all_headers.extend(headers)
            # Remove duplicates while preserving order
            unique_headers = list(dict.fromkeys(all_headers))
        except Exception as e:
            messages.error(request, f"Error processing Excel file: {e}")
            return redirect('dashboard')

        # Initialize the form with the extracted headers
        form = AxesSelectionForm(headers=unique_headers)
        return render(request, 'NanoAnalyst/select-axis.html', {'form': form, 'file': uploaded_file})


@method_decorator(login_required, name='dispatch')
class AdvancedPlotView(View):
    template_name = 'NanoAnalyst/advanced_plot.html'

    def get(self, request):
        # Define the list of column names manually
        column_names = [
            'Displacement Into Surface (nm)',
            'Load On Sample (mN)',
            'Time On Sample (s)',
            'Harmonic Contact Stiffness (N/m)',
            'Hardness (GPa)',
            'Modulus (GPa)',
        ]
        # Render the form with the column names
        return render(request, self.template_name, {'column_names': column_names})

    def post(self, request):
        # Get the uploaded file
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return render(request, self.template_name, {'error': 'Please upload a file.'})

        # Define the list of column names manually
        column_names = [
            'Displacement Into Surface (nm)',
            'Load On Sample (mN)',
            'Time On Sample (s)',
            'Harmonic Contact Stiffness (N/m)',
            'Hardness (GPa)',
            'Modulus (GPa)',
        ]

        # Get user inputs
        x_axis_column = request.POST.get('x_axis_column')
        y_axis_column = request.POST.get('y_axis_column')
        plot_title = request.POST.get('plot_title', '')
        x_axis_label = request.POST.get('x_axis_label', '')
        y_axis_label = request.POST.get('y_axis_label', '')

        # Get optional axis ranges
        x_min = request.POST.get('x_min')
        x_max = request.POST.get('x_max')
        y_min = request.POST.get('y_min')
        y_max = request.POST.get('y_max')

        # Validate x and y axis columns
        if not x_axis_column or not y_axis_column:
            return render(request, self.template_name, {
                'error': 'Please select both x and y axis columns.',
                'column_names': column_names,
            })

        # Generate sheet names
        sheet_names = [f"Test {str(i).zfill(3)}" for i in range(0, 21)]

        # Initialize an empty DataFrame
        combined_df = pd.DataFrame()

        test_numbers = np.array([int(s.split()[-1]) for s in sheet_names])

        # Loop through sheets and process
        for i, sheet_name in zip(test_numbers, sheet_names):
            try:
                # Load the current sheet as strings
                df_loaded = pd.read_excel(uploaded_file, sheet_name=sheet_name, dtype=str, engine='xlrd')

                # Rename columns
                df_loaded.columns = [
                    'Segment',
                    f'Displacement Into Surface (nm) - {i}',
                    f'Load On Sample (mN) - {i}',
                    f'Time On Sample (s) - {i}',
                    f'Harmonic Contact Stiffness (N/m) - {i}',
                    f'Hardness (GPa) - {i}',
                    f'Modulus (GPa) - {i}'
                ]

                # Retain only the renamed columns (excluding 'Segment' for consistency)
                df_loaded = df_loaded.drop(columns=['Segment'])

                # Combine with the main DataFrame
                if combined_df.empty:
                    combined_df = df_loaded
                else:
                    combined_df = pd.concat([combined_df, df_loaded], axis=1)
            except:
                pass

        # Extract all x and y columns
        x_columns = [col for col in combined_df.columns if x_axis_column in col]
        y_columns = [col for col in combined_df.columns if y_axis_column in col]

        # Generate the combined plot
        plt.figure(figsize=(10, 6))
        for x_col, y_col in zip(x_columns, y_columns):
            try:
                # Convert to numeric, skipping invalid entries
                x_data = pd.to_numeric(combined_df[x_col], errors='coerce')
                y_data = pd.to_numeric(combined_df[y_col], errors='coerce')

                # Plot valid data
                plt.plot(x_data, y_data, label=f"{x_col.split('-')[-1].strip()}")
            except:
                pass

        # Calculate mean and standard deviation
        mean_y_data = np.mean([pd.to_numeric(combined_df[y_col], errors='coerce') for y_col in y_columns], axis=0)
        std_y_data = np.std([pd.to_numeric(combined_df[y_col], errors='coerce') for y_col in y_columns], axis=0)

        # Plot mean with standard deviation
        plt.plot(pd.to_numeric(combined_df[x_columns[0]], errors='coerce'), mean_y_data, label='Mean', color='black')
        plt.fill_between(pd.to_numeric(combined_df[x_columns[0]], errors='coerce'), mean_y_data - std_y_data, mean_y_data + std_y_data, alpha=0.6, label='Standard Deviation')

        # Customize the plot
        plt.title(plot_title or f"{x_axis_column} vs {y_axis_column}", fontsize=14)
        plt.xlabel(x_axis_label or x_axis_column, fontsize=12)
        plt.ylabel(y_axis_label or y_axis_column, fontsize=12)
        plt.legend(title="Test Number", loc="best")
        plt.grid(True)
        plt.tight_layout()

        # Apply optional axis ranges
        if x_min and x_max:
            plt.xlim(float(x_min), float(x_max))
        if y_min and y_max:
            plt.ylim(float(y_min), float(y_max))

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Render the template with the combined plot and form data
        return render(request, self.template_name, {
            'plot': plot_base64,
            'column_names': column_names,
            'form_data': {
                'x_axis_column': x_axis_column,
                'y_axis_column': y_axis_column,
                'plot_title': plot_title,
                'x_axis_label': x_axis_label,
                'y_axis_label': y_axis_label,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
            },
        })