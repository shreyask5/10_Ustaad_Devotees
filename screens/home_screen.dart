import 'package:flutter/material.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Women Safety App'),
        centerTitle: true,
      ),
      body: GridView.count(
        crossAxisCount: 2,
        padding: const EdgeInsets.all(16.0),
        mainAxisSpacing: 16.0,
        crossAxisSpacing: 16.0,
        children: [
          _buildActionCard(
            context,
            'SOS',
            Icons.warning_rounded,
            Colors.red,
            '/sos',
          ),
          _buildActionCard(
            context,
            'Safety Map',
            Icons.map_rounded,
            Colors.blue,
            '/safety-map',
          ),
          _buildActionCard(
            context,
            'Guidance',
            Icons.book_rounded,
            Colors.green,
            '/guidance',
          ),
          _buildActionCard(
            context,
            'Emergency Contacts',
            Icons.contacts_rounded,
            Colors.orange,
            '/contacts',
          ),
        ],
      ),
    );
  }

  Widget _buildActionCard(BuildContext context, String title, IconData icon,
      Color color, String route) {
    return Card(
      elevation: 4.0,
      child: InkWell(
        onTap: () => Navigator.pushNamed(context, route),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 48.0,
              color: color,
            ),
            const SizedBox(height: 8.0),
            Text(
              title,
              style: const TextStyle(
                fontSize: 18.0,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
} 