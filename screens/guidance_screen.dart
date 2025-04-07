import 'package:flutter/material.dart';
import '../services/guidance_service.dart';

class GuidanceScreen extends StatefulWidget {
  const GuidanceScreen({super.key});

  @override
  State<GuidanceScreen> createState() => _GuidanceScreenState();
}

class _GuidanceScreenState extends State<GuidanceScreen> {
  final GuidanceService _guidanceService = GuidanceService();
  bool _isLoading = true;
  List<GuidanceItem> _guidanceItems = [];

  @override
  void initState() {
    super.initState();
    _loadGuidanceContent();
  }

  Future<void> _loadGuidanceContent() async {
    try {
      final items = await _guidanceService.getGuidanceContent();
      setState(() {
        _guidanceItems = items;
        _isLoading = false;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error loading guidance content: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Safety Guidance'),
        centerTitle: true,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : ListView.builder(
              padding: const EdgeInsets.all(16.0),
              itemCount: _guidanceItems.length,
              itemBuilder: (context, index) {
                final item = _guidanceItems[index];
                return Card(
                  margin: const EdgeInsets.only(bottom: 16.0),
                  child: ExpansionTile(
                    title: Text(
                      item.title,
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 18.0,
                      ),
                    ),
                    subtitle: Text(item.subtitle),
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              item.content,
                              style: const TextStyle(fontSize: 16.0),
                            ),
                            if (item.videoUrl != null) ...[
                              const SizedBox(height: 16.0),
                              ElevatedButton.icon(
                                onPressed: () {
                                  // Implement video playback
                                },
                                icon: const Icon(Icons.play_circle_outline),
                                label: const Text('Watch Video'),
                              ),
                            ],
                            if (item.emergencyContacts.isNotEmpty) ...[
                              const SizedBox(height: 16.0),
                              const Text(
                                'Emergency Contacts:',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16.0,
                                ),
                              ),
                              const SizedBox(height: 8.0),
                              ...item.emergencyContacts.map(
                                (contact) => ListTile(
                                  leading: const Icon(Icons.phone),
                                  title: Text(contact.name),
                                  subtitle: Text(contact.number),
                                  trailing: IconButton(
                                    icon: const Icon(Icons.call),
                                    onPressed: () {
                                      // Implement call functionality
                                    },
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
    );
  }
} 