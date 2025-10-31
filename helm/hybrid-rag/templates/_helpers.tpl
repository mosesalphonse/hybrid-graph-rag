{{/* Common labels */}}
{{- define "hybrid-rag.labels" -}}
helm.sh/chart: {{ include "hybrid-rag.chart" . }}
{{ include "hybrid-rag.selectorLabels" . }}
{{- end }}

{{/* Selector labels */}}
{{- define "hybrid-rag.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hybrid-rag.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/* Chart name */}}
{{- define "hybrid-rag.chart" -}}
{{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end }}

{{/* App name */}}
{{- define "hybrid-rag.name" -}}
{{ .Chart.Name }}
{{- end }}

{{/* Full name */}}
{{- define "hybrid-rag.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}