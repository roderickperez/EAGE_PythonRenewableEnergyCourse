# Folium

Folium creates Leaflet web maps. Coordinates are normally supplied as latitude–longitude pairs in WGS84.

```python
import folium

energy_map = folium.Map(location=[48.2, 16.37], zoom_start=5)
folium.Marker([48.2, 16.37], tooltip="Example site").add_to(energy_map)
energy_map
```

Avoid exposing confidential infrastructure coordinates. For dense point data, use clustering or aggregation rather than thousands of overlapping markers.
